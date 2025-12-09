
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from hand_tracking.pipeline import HandTrackingPipeline

try:
    print("Initializing HandTrackingPipeline...")
    pipeline = HandTrackingPipeline(prune_ratio=0.0, precision="fp16")
    print("Success!")
except Exception as e:
    print(f"Caught exception: {e}")
    import traceback
    traceback.print_exc()
