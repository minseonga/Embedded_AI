
import torch
from torch import nn, optim
import onnx
import onnxruntime as ort
import numpy as np
from onnx2torch import convert
from onnx2torch.node_converters.registry import _CONVERTER_REGISTRY, OperationDescription
from onnx2torch.utils.common import OperationConverterResult, OnnxMapping
import time
import copy

# --- Custom Converters (reused from verify_onnx2torch.py) ---
# We need these to ensure the model structure is at least loadable

def convert_pad(node, graph, *args, **kwargs):
    mode = "constant"
    if hasattr(node, 'attributes'):
        mode = node.attributes.get('mode', 'constant')
        if isinstance(mode, bytes):
            mode = mode.decode('utf-8')
    elif hasattr(node, 'pb'):
        for attr in node.pb.attribute:
            if attr.name == "mode":
                mode = attr.s.decode("utf-8")
    if mode == "edge":
        mode = "replicate"
    
    class OnnxPad(nn.Module):
        def __init__(self, mode='constant'):
            super().__init__()
            self.mode = mode
        def forward(self, x, pads=None, constant_value=0.0, axes=None):
            if pads is None: return x
            if isinstance(pads, torch.Tensor):
                pads_list = pads.detach().cpu().numpy().tolist()
            else:
                pads_list = pads
            rank = x.dim()
            if len(pads_list) < 2 * rank: return x
            torch_pads = []
            for i in range(rank - 1, -1, -1):
                pad_begin = int(pads_list[i])
                pad_end = int(pads_list[i + rank])
                torch_pads.extend([pad_begin, pad_end])
            val = constant_value
            if isinstance(val, torch.Tensor): val = val.item()
            return torch.nn.functional.pad(x, torch_pads, mode=self.mode, value=val)

    return OperationConverterResult(
        torch_module=OnnxPad(mode=mode),
        onnx_mapping=OnnxMapping(inputs=tuple(node.input_values), outputs=tuple(node.output_values)),
    )

def convert_reshape(node, graph, *args, **kwargs):
    class OnnxReshape(nn.Module):
        def forward(self, x, shape):
            if isinstance(shape, torch.Tensor):
                shape = shape.detach().cpu().numpy().tolist()
                shape = [int(s) for s in shape]
            new_shape = []
            for i, s in enumerate(shape):
                if s == 0:
                    if i < x.dim(): new_shape.append(x.shape[i])
                    else: new_shape.append(0) 
                else: new_shape.append(s)
            return torch.reshape(x, new_shape)

    return OperationConverterResult(
        torch_module=OnnxReshape(),
        onnx_mapping=OnnxMapping(inputs=tuple(node.input_values), outputs=tuple(node.output_values)),
    )

# Register converters
from onnx2torch.node_converters.registry import add_converter
add_converter(operation_type="Pad", version=18)(convert_pad)
for version in [5, 13, 14, 19]:
    desc = OperationDescription(domain='', operation_type='Reshape', version=version)
    _CONVERTER_REGISTRY[desc] = convert_reshape

def fix_weights():
    onnx_path = "assets/models/palm_detector_fp16.onnx"
    print(f"Loading {onnx_path}...")
    
    # 1. Load Teacher (ONNX)
    ort_session = ort.InferenceSession(onnx_path)
    
    # 2. Load Student (PyTorch)
    try:
        model = convert(onnx_path)
    except Exception as e:
        print(f"Conversion failed: {e}")
        return

    model.train() # Enable gradient computation
    # But we want to keep BatchNorm stats frozen? 
    # Usually for weight correction we want to tune everything or just weights.
    # Let's keep it in train mode but maybe freeze BN running stats if needed.
    # For now, just train mode.
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = nn.L1Loss()
    
    print("Starting Weight Correction (Distillation)...")
    
    for step in range(200):
        # Generate random input
        # Use random noise in range [-1, 1]
        dummy_input = torch.rand(1, 3, 256, 256) * 2 - 1
        dummy_np = dummy_input.numpy().astype(np.float32)
        
        # Teacher Output
        ort_outs = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_np})
        target_outs = [torch.from_numpy(o) for o in ort_outs]
        
        # Student Output
        optimizer.zero_grad()
        student_outs = model(dummy_input)
        
        if not isinstance(student_outs, (list, tuple)):
            student_outs = [student_outs]
            
        loss = 0
        for s_out, t_out in zip(student_outs, target_outs):
            loss += criterion(s_out, t_out)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")
            scheduler.step(loss)
            
        if loss.item() < 1e-5:
            print("Converged!")
            break
            
    # Verification
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 256, 256)
        dummy_np = dummy_input.numpy().astype(np.float32)
        ort_outs = ort_session.run(None, {ort_session.get_inputs()[0].name: dummy_np})
        student_outs = model(dummy_input)
        if not isinstance(student_outs, (list, tuple)):
            student_outs = [student_outs]
            
        max_diff = 0
        for s_out, o_out in zip(student_outs, ort_outs):
            diff = np.abs(s_out.numpy() - o_out).max()
            max_diff = max(max_diff, diff)
            
    print(f"Final Max Diff: {max_diff:.6f}")
    
    if max_diff < 0.1: # Threshold for success
        save_path = "assets/models/palm_detector.pth"
        torch.save(model, save_path) # Save entire model (easier for loading)
        print(f"Saved corrected model to {save_path}")
    else:
        print("Failed to correct weights sufficiently.")

if __name__ == "__main__":
    fix_weights()
