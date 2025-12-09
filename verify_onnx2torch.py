
import torch
from torch import nn
import onnx
import onnxruntime as ort
import numpy as np
from onnx2torch import convert
from onnx2torch.utils.common import OperationConverterResult, OnnxMapping
from onnx2torch.node_converters.registry import add_converter
import time

# Custom Pad converter for Pad-18
@add_converter(operation_type="Pad", version=18)
def convert_pad(node, graph, *args, **kwargs):
    # node is onnx2torch.onnx_node.OnnxNode
    
    # Get mode from attributes
    mode = "constant"
    if hasattr(node, 'attributes'):
        mode = node.attributes.get('mode', 'constant')
        if isinstance(mode, bytes):
            mode = mode.decode('utf-8')
    elif hasattr(node, 'pb'):
        for attr in node.pb.attribute:
            if attr.name == "mode":
                mode = attr.s.decode("utf-8")
    
    # Map ONNX mode to PyTorch mode
    if mode == "edge":
        mode = "replicate"
    
    class OnnxPad(nn.Module):
        def __init__(self, mode='constant'):
            super().__init__()
            self.mode = mode
            
        def forward(self, x, pads=None, constant_value=0.0, axes=None):
            if pads is None:
                return x
            
            # pads: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
            # PyTorch F.pad: [dimN_begin, dimN_end, dimN-1_begin, dimN-1_end, ...]
            
            if isinstance(pads, torch.Tensor):
                pads_list = pads.detach().cpu().numpy().tolist()
            else:
                pads_list = pads
                
            rank = x.dim()
            if len(pads_list) < 2 * rank:
                return x
                
            torch_pads = []
            # Iterate from last dim (R-1) down to 0
            for i in range(rank - 1, -1, -1):
                pad_begin = int(pads_list[i])
                pad_end = int(pads_list[i + rank])
                torch_pads.extend([pad_begin, pad_end])
            
            val = constant_value
            if isinstance(val, torch.Tensor):
                val = val.item()
                
            return torch.nn.functional.pad(x, torch_pads, mode=self.mode, value=val)

    return OperationConverterResult(
        torch_module=OnnxPad(mode=mode),
        onnx_mapping=OnnxMapping(
            inputs=tuple(node.input_values),
            outputs=tuple(node.output_values),
        ),
    )

from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription

def convert_reshape(node, graph, *args, **kwargs):
    class OnnxReshape(nn.Module):
        def forward(self, x, shape):
            if isinstance(shape, torch.Tensor):
                shape = shape.detach().cpu().numpy().tolist()
                shape = [int(s) for s in shape]
            
            new_shape = []
            for i, s in enumerate(shape):
                if s == 0:
                    # ONNX: 0 means copy from input dimension
                    if i < x.dim():
                        new_shape.append(x.shape[i])
                    else:
                        new_shape.append(0) 
                else:
                    new_shape.append(s)
            
            return torch.reshape(x, new_shape)

    return OperationConverterResult(
        torch_module=OnnxReshape(),
        onnx_mapping=OnnxMapping(
            inputs=tuple(node.input_values),
            outputs=tuple(node.output_values),
        ),
    )

# Manually register/overwrite Reshape converters
for version in [5, 13, 14, 19]:
    desc = OperationDescription(domain='', operation_type='Reshape', version=version)
    _CONVERTER_REGISTRY[desc] = convert_reshape

def verify_conversion():
    onnx_path = "assets/models/palm_detector_fp16.onnx"
    print(f"Loading {onnx_path}...")
    
    # 1. Convert using onnx2torch
    start = time.time()
    try:
        model = convert(onnx_path)
        print(f"Conversion successful in {time.time() - start:.4f}s")
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return

    model.eval()
    
    # Experiment: Toggle align_corners for Upsample layers
    print("Experiment: Forcing align_corners=False for all Upsample layers...")
    for m in model.modules():
        if isinstance(m, nn.Upsample):
            print(f"  Found Upsample: mode={m.mode}, align_corners={m.align_corners}")
            # Force align_corners=False (or True if it was False)
            # Usually ONNX 'linear' with 'half_pixel' maps to align_corners=False
            # But let's try to flip it to see if it helps.
            # m.align_corners = not m.align_corners
            m.align_corners = True
            # Note: align_corners is only valid for linear/bicubic/trilinear
            if m.mode in ('nearest', 'area'):
                m.align_corners = None
    
    # 2. ONNX Runtime (Reference)
    ort_session = ort.InferenceSession(onnx_path)
    
    # Input
    dummy_input = torch.randn(1, 3, 256, 256)
    dummy_np = dummy_input.numpy().astype(np.float32)
    
    # Run ORT
    ort_outs = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_np})
    
    # Run PyTorch
    with torch.no_grad():
        torch_out = model(dummy_input)
    
    # Hook intermediate layers
    print("\nIntermediate Layer Comparison:")
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
        # Hook Conv, ConvTranspose, MaxPool, Upsample/Resize, and our custom modules
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.MaxPool2d, torch.nn.Upsample)):
            layer.register_forward_hook(get_activation(name))
        # Also hook our custom modules if possible (they are closures, so hard to name?)
        # onnx2torch names modules like 'Node_name'
        if "OnnxPad" in str(type(layer)) or "OnnxReshape" in str(type(layer)):
             layer.register_forward_hook(get_activation(name))
            
    # Run PyTorch again
    with torch.no_grad():
        model(dummy_input)
        
    for name, act in activations.items():
        if isinstance(act, (list, tuple)):
            for i, a in enumerate(act):
                if isinstance(a, torch.Tensor):
                    if a.dtype in (torch.long, torch.int32, torch.int64):
                        a_float = a.float()
                        print(f"{name}[{i}]: shape={a.shape}, min={a_float.min():.4f}, max={a_float.max():.4f}, mean={a_float.mean():.4f} (Int)")
                    else:
                        print(f"{name}[{i}]: shape={a.shape}, min={a.min():.4f}, max={a.max():.4f}, mean={a.mean():.4f}")
        elif isinstance(act, torch.Tensor):
            if act.dtype in (torch.long, torch.int32, torch.int64):
                a_float = act.float()
                print(f"{name}: shape={act.shape}, min={a_float.min():.4f}, max={a_float.max():.4f}, mean={a_float.mean():.4f} (Int)")
            else:
                print(f"{name}: shape={act.shape}, min={act.min():.4f}, max={act.max():.4f}, mean={act.mean():.4f}")

if __name__ == "__main__":
    verify_conversion()
