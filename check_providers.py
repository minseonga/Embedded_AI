
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from shared.ort_wrapper import OrtWrapper

def check():
    print("Checking OrtWrapper providers...")
    onnx_path = Path("assets/models/palm_detector_fp16.onnx")
    
    # Check CPU
    print("\n--- Testing CPU ---")
    try:
        wrapper = OrtWrapper(onnx_path, torch.device("cpu"))
    except Exception as e:
        print(f"Failed to init CPU wrapper: {e}")

    # Check CUDA (if available or requested)
    print("\n--- Testing CUDA Request ---")
    try:
        wrapper = OrtWrapper(onnx_path, torch.device("cuda"))
    except Exception as e:
        print(f"Failed to init CUDA wrapper (expected if no GPU): {e}")
        
    # Check MPS (Mac)
    print("\n--- Testing MPS Request ---")
    try:
        wrapper = OrtWrapper(onnx_path, torch.device("mps"))
    except Exception as e:
        print(f"Failed to init MPS wrapper (expected if no Mac): {e}")

if __name__ == "__main__":
    check()
