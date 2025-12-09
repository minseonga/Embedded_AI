
import sys
import os
import torch
import onnxruntime as ort
import numpy as np
sys.path.append(os.path.join(os.getcwd(), "src"))

from face_landmark.pipeline import TorchOnnxWrapper

def compare_outputs(onnx_path):
    print(f"Checking {os.path.basename(onnx_path)}...")
    
    # 1. Run ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    # Handle dynamic batch size if needed, assume 1 for now
    input_shape = [1 if isinstance(d, str) else d for d in input_shape]
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    ort_inputs = {input_name: dummy_input}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # 2. Run onnx2pytorch
    try:
        wrapper = TorchOnnxWrapper(onnx_path, device=torch.device("cpu"), precision="fp32")
        torch_input = torch.from_numpy(dummy_input)
        with torch.no_grad():
            torch_out = wrapper(torch_input)
            
        # Handle multiple outputs
        if isinstance(torch_out, (list, tuple)):
            torch_outs = [t.numpy() for t in torch_out]
        elif isinstance(torch_out, dict):
             # Try to match by order or name if possible, but for now just list values
             torch_outs = [v.numpy() for v in torch_out.values()]
        else:
            torch_outs = [torch_out.numpy()]
            
        print(f"ONNX Runtime outputs: {len(ort_outs)}")
        print(f"PyTorch outputs: {len(torch_outs)}")
        
        for i, (o_ort, o_torch) in enumerate(zip(ort_outs, torch_outs)):
            print(f"Output {i}:")
            print(f"  ORT shape: {o_ort.shape}")
            print(f"  PT  shape: {o_torch.shape}")
            diff = np.abs(o_ort - o_torch)
            print(f"  Max diff: {diff.max():.6f}")
            print(f"  Mean diff: {diff.mean():.6f}")
            if diff.max() > 1e-3:
                print("  [WARNING] Significant difference!")
            else:
                print("  [OK] Matches well.")
                
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_outputs("assets/models/ultraface_rfb_320_simplified.onnx")
    compare_outputs("assets/models/palm_detector_fp16.onnx")
