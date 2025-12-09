
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from face_landmark.pipeline import FaceLandmarkPipeline

try:
    print("Initializing FaceLandmarkPipeline...")
    # Use fp16 as in the user's command (though user used int8 which falls back to fp16)
    pipeline = FaceLandmarkPipeline(precision="fp16", use_cuda=False)
    print("Pipeline initialized successfully.")

    # Run inference on dummy frame
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    print("Running inference on dummy frame...")
    landmarks, mar, face_box = pipeline.process_frame(dummy_frame)
    print(f"Inference successful. Face box: {face_box}")
    
except Exception as e:
    print(f"Caught exception: {e}")
    import traceback
    traceback.print_exc()
