
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from hand_tracking.pipeline import HandTrackingPipeline

def test_config(prune, precision):
    print(f"\n--- Testing Prune={prune}, Precision={precision} ---")
    try:
        pipeline = HandTrackingPipeline(prune_ratio=prune, precision=precision)
        pipeline.print_stats()
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

print("Verifying Pruning and Quantization...")
test_config(0.0, "fp16")
test_config(0.3, "fp16")
test_config(0.0, "fp32")
