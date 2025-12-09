
import torch
import onnx
import onnxruntime as ort
import numpy as np
from onnx2pytorch import ConvertModel
from onnx import helper as onnx_helper

class TorchOnnxWrapper:
    def __init__(self, onnx_path):
        onnx_model = onnx.load(str(onnx_path))
        onnx_model = self._patch_conv_kernel_size(onnx_model)
        onnx_model = self._remove_storage_order(onnx_model)
        
        self.model = ConvertModel(onnx_model, experimental=True)
        self.model.eval()

    @staticmethod
    def _patch_conv_kernel_size(onnx_model):
        init_map = {init.name: init for init in onnx_model.graph.initializer}
        for node in onnx_model.graph.node:
            if node.op_type not in ("Conv", "ConvTranspose"):
                continue
            attr_names = {attr.name for attr in node.attribute}
            if "kernel_size" in attr_names:
                continue
            kernel_shape_attr = next((a for a in node.attribute if a.name == "kernel_shape"), None)
            kernel_shape = list(kernel_shape_attr.ints) if kernel_shape_attr is not None else None

            if kernel_shape is None and len(node.input) > 1:
                weight_name = node.input[1]
                weight_init = init_map.get(weight_name)
                if weight_init and len(weight_init.dims) >= 4:
                    kernel_shape = [int(weight_init.dims[2]), int(weight_init.dims[3])]
                    node.attribute.append(onnx_helper.make_attribute("kernel_shape", kernel_shape))
        return onnx_model

    @staticmethod
    def _remove_storage_order(onnx_model):
        for node in onnx_model.graph.node:
            for i, attr in enumerate(node.attribute):
                if attr.name == "storage_order":
                    node.attribute.pop(i)
                    break
        return onnx_model

def compare_outputs():
    onnx_path = "assets/models/palm_detector_fp16.onnx"
    print(f"Loading {onnx_path}...")
    
    # 1. PyTorch Model
    wrapper = TorchOnnxWrapper(onnx_path)
    model = wrapper.model
    
    # 2. ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    
    # Input
    dummy_input = torch.randn(1, 3, 256, 256)
    dummy_np = dummy_input.numpy().astype(np.float32) # ORT expects float32 usually
    
    # Run ORT
    ort_outs = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_np})
    
    # Run PyTorch
    with torch.no_grad():
        torch_out = model(dummy_input)
    
    # Compare final output
    print("\nFinal Output Comparison:")
    if isinstance(torch_out, (list, tuple)):
        for i, (to, oo) in enumerate(zip(torch_out, ort_outs)):
            diff = np.abs(to.numpy() - oo).max()
            print(f"Output {i}: Max Diff = {diff:.6f}")
    else:
        diff = np.abs(torch_out.numpy() - ort_outs[0]).max()
        print(f"Output 0: Max Diff = {diff:.6f}")
        
    # Hook intermediate layers
    print("\nIntermediate Layer Comparison:")
    # We can't easily match ORT nodes to PyTorch modules because onnx2pytorch structure is complex.
    # But we can print PyTorch module names and their output stats.
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, (list, tuple)):
                activations[name] = [o.detach() if isinstance(o, torch.Tensor) else o for o in output]
            elif isinstance(output, torch.Tensor):
                activations[name] = output.detach()
            else:
                activations[name] = output
        return hook

    for name, layer in model.named_modules():
        # Hook Conv, ConvTranspose, MaxPool, Upsample/Resize
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.MaxPool2d, torch.nn.Upsample)):
            layer.register_forward_hook(get_activation(name))
            
    # Run PyTorch again
    with torch.no_grad():
        model(dummy_input)
        
    # We don't have intermediate ORT outputs easily without modifying the model to output them.
    # But we can check if activations look "sane" or if they explode.
    for name, act in activations.items():
        if isinstance(act, (list, tuple)):
            for i, a in enumerate(act):
                if isinstance(a, torch.Tensor):
                    if a.dtype in (torch.long, torch.int32, torch.int64):
                        a_float = a.float()
                        print(f"{name}[{i}]: shape={a.shape}, min={a_float.min():.4f}, max={a_float.max():.4f}, mean={a_float.mean():.4f} (Int/Long)")
                    else:
                        print(f"{name}[{i}]: shape={a.shape}, min={a.min():.4f}, max={a.max():.4f}, mean={a.mean():.4f}")
                else:
                    print(f"{name}[{i}]: type={type(a)}")
        else:
            print(f"{name}: shape={act.shape}, min={act.min():.4f}, max={act.max():.4f}, mean={act.mean():.4f}")

if __name__ == "__main__":
    compare_outputs()
