
import cv2
import numpy as np
import torch
import time
import sys
from src.hand_tracking import BlazeHandTrackingPipeline

def test_device(device_name):
    print(f"\n[{device_name.upper()}] Initializing Pipeline...")
    try:
        pipeline = BlazeHandTrackingPipeline(
            device=device_name,
            prune_ratio=0.3,
            quantize=False,
            precision='fp32'
        )
        print(f"[{device_name.upper()}] Pipeline initialized.")
    except Exception as e:
        print(f"[{device_name.upper()}] Initialization failed: {e}")
        return

    print(f"[{device_name.upper()}] Running Inference (Dummy Image)...")
    dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
    
    try:
        start = time.time()
        # Warmup
        print(f"[{device_name.upper()}] Sending frame to device...")
        landmarks, detections, scale, pad = pipeline.process_frame(dummy_frame)
        torch.cuda.synchronize() if device_name == 'cuda' else None
        elapsed = (time.time() - start) * 1000
        
        print(f"[{device_name.upper()}] Inference finished in {elapsed:.2f} ms")
        print(f"[{device_name.upper()}] Detections: {len(detections)}")
        print(f"[{device_name.upper()}] PASS")
    except Exception as e:
        print(f"[{device_name.upper()}] FAIL: {e}")
        import traceback
        traceback.print_exc()

def test_pipeline():
    print("="*60)
    print("Diagnostics: BlazeHandTrackingPipeline")
    print("="*60)
    
    # 1. Test CPU first (Baseline)
    test_device('cpu')
    
    # 2. Test CUDA if available
    if torch.cuda.is_available():
        test_device('cuda')
    else:
        print("\n[CUDA] Skipped (Not available)")

if __name__ == "__main__":
    test_pipeline()
