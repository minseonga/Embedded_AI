
import cv2
import numpy as np
import torch
import time
from src.hand_tracking import BlazeHandTrackingPipeline
from src.hand_tracking.blazebase import resize_pad

class DebugPipeline(BlazeHandTrackingPipeline):
    def process_frame_debug(self, frame):
        print("\n[DEBUG] Processing Frame...")
        
        # 1. Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img1, img2, scale, pad = resize_pad(frame_rgb)
        
        # Save input to detector
        cv2.imwrite("debug_input_256.jpg", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        print(f"[DEBUG] Saved debug_input_256.jpg (Shape: {img1.shape})")
        
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        img1_tensor = img1_tensor.to(self.device)
        
        if self.precision == 'fp16' and self.device.type == 'cuda':
            img1_tensor = img1_tensor.half()
            
        # 2. Run Detector Raw
        with torch.no_grad():
            # Call model directly to get raw output
            out = self.detector(img1_tensor)
            # out is (classification, regression)
            raw_scores = out[0] # (B, 896, 1) or similar
            raw_boxes = out[1]
            
            # Check scores
            scores = raw_scores.sigmoid().cpu().numpy()
            print(f"[DEBUG] Raw Scores - Min: {scores.min():.4f}, Max: {scores.max():.4f}, Mean: {scores.mean():.4f}")
            
            count_over_05 = (scores > 0.5).sum()
            count_over_01 = (scores > 0.1).sum()
            print(f"[DEBUG] Candidates > 0.5: {count_over_05}")
            print(f"[DEBUG] Candidates > 0.1: {count_over_01}")
            
            # Run standard prediction
            detections = self.detector.predict_on_batch(img1_tensor)[0]
            
        print(f"[DEBUG] Detections after NMS: {len(detections)}")
        
        # Draw detections on debug image
        debug_img = img1.copy()
        if len(detections) > 0:
            for i in range(len(detections)):
                det = detections[i]
                # det is (ymin, xmin, ymax, xmax, ...) relative to 256x256?
                # No, predict_on_batch returns pixel coords in 256x256 because we fixed _decode_boxes?
                # Wait, let's check blazepalm.py _decode_boxes again.
                # It uses anchors. Anchors are usually normalized [0,1].
                # If anchors are [0,1], output is [0,1].
                # Let's assume [0,1] for now and draw.
                
                ymin, xmin, ymax, xmax = det[:4].cpu().numpy()
                h, w = debug_img.shape[:2]
                
                # Try drawing as normalized
                p1 = (int(xmin * w), int(ymin * h))
                p2 = (int(xmax * w), int(ymax * h))
                cv2.rectangle(debug_img, p1, p2, (0, 255, 0), 2)
                
                # Try drawing as pixels (if it was pixels)
                p1_px = (int(xmin), int(ymin))
                p2_px = (int(xmax), int(ymax))
                cv2.rectangle(debug_img, p1_px, p2_px, (0, 0, 255), 2)
                
        cv2.imwrite("debug_detection.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        print("[DEBUG] Saved debug_detection.jpg (Green=Normalized, Red=Pixel)")
        
        return [], [], scale, pad

def main():
    pipeline = DebugPipeline(precision='fp32')
    
    # Try capturing from camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Cannot open camera, using dummy.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        print("Capturing frame...")
        # Read a few frames to settle auto-exposure
        for _ in range(30):
            ret, frame = cap.read()
        
        if not ret:
            print("Failed to read frame.")
            return
            
    cv2.imwrite("debug_original.jpg", frame)
    print("[DEBUG] Saved debug_original.jpg")
    
    pipeline.process_frame_debug(frame)

if __name__ == "__main__":
    main()
