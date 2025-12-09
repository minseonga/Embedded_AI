
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
        # Warmup
        print(f"[{device_name.upper()}] Warmup (1/3)...")
        for _ in range(3):
            pipeline.process_frame(dummy_frame)
            if device_name == 'cuda': torch.cuda.synchronize()
        
        # Benchmark
        print(f"[{device_name.upper()}] Benchmarking (50 runs)...")
        start = time.time()
        for _ in range(50):
            pipeline.process_frame(dummy_frame)
            if device_name == 'cuda': torch.cuda.synchronize()
        total_time = time.time() - start
        avg_ms = (total_time / 50) * 1000
        fps = 50 / total_time
        
        print(f"[{device_name.upper()}] Average Inference: {avg_ms:.2f} ms ({fps:.2f} FPS)")
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
