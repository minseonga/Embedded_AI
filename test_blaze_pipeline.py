
import cv2
import numpy as np
import torch
import time
from src.hand_tracking import BlazeHandTrackingPipeline, draw_detections, draw_landmarks

def test_pipeline():
    print("="*60)
    print("Testing BlazeHandTrackingPipeline (Pure PyTorch)")
    print("="*60)
    
    # 1. Initialize Pipeline with Pruning and Quantization
    print("[1] Initializing Pipeline...")
    try:
        pipeline = BlazeHandTrackingPipeline(
            prune_ratio=0.3,  # Request 30% pruning
            quantize=False,   # Dynamic quantization (CPU only usually)
            precision='fp32'  # Use fp32 for safety first
        )
        print("    Pipeline initialized successfully.")
    except Exception as e:
        print(f"    Failed to initialize: {e}")
        return

    # 2. Check Pruning
    print("\n[2] Verifying Pruning...")
    total_zeros = 0
    total_elements = 0
    for name, module in pipeline.detector.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            total_zeros += torch.sum(module.weight == 0)
            total_elements += module.weight.nelement()
    
    sparsity = 100. * total_zeros / total_elements
    print(f"    Detector Sparsity: {sparsity:.2f}%")
    if sparsity > 0:
        print("    [PASS] Pruning applied successfully.")
    else:
        print("    [FAIL] Pruning not applied.")

    # 3. Run Inference on Dummy Image
    print("\n[3] Running Inference (Dummy Image)...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a fake hand (white rectangle) to trigger detection? 
    # Unlikely to work with trained model, but we check for crashes.
    cv2.rectangle(dummy_frame, (200, 200), (400, 400), (255, 255, 255), -1)
    
    try:
        start = time.time()
        landmarks, detections, scale, pad = pipeline.process_frame(dummy_frame)
        elapsed = (time.time() - start) * 1000
        print(f"    Inference time: {elapsed:.2f} ms")
        print(f"    Detections: {len(detections)}")
        print("    [PASS] Inference ran without crash.")
    except Exception as e:
        print(f"    [FAIL] Inference failed: {e}")
        import traceback
        traceback.print_exc()

    # 4. Webcam Loop (Optional, if running interactively)
    # print("\n[4] Starting Webcam Demo (Press ESC to quit)...")
    # pipeline.run_webcam() # We haven't implemented run_webcam in BlazePipeline yet

if __name__ == "__main__":
    test_pipeline()
