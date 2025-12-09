#!/usr/bin/env python3
"""
Quick test for MediaPipe Face Detection.
Run this to verify face detection is working.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from face_landmark import FaceLandmarkPipeline, draw_face_box

print("=" * 60)
print("MediaPipe Face Detection Test")
print("=" * 60)

# Initialize pipeline
pipeline = FaceLandmarkPipeline(precision="fp16")
pipeline.print_stats()

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)

print("\nCamera opened successfully")
print("Testing face detection...")
print("Press 'q' or ESC to quit")
print("=" * 60)

frame_count = 0
detected_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        frame_count += 1

        # Process frame
        landmarks, mar, face_box = pipeline.process_frame(frame)

        # Draw results
        if face_box is not None:
            detected_count += 1
            draw_face_box(frame, face_box, color=(0, 255, 0))

            # Draw mouth center
            mouth_center = pipeline.get_mouth_center(None, face_box)
            if mouth_center is not None:
                cv2.circle(frame, tuple(mouth_center.astype(int)), 5, (0, 255, 255), -1)

            # Show MAR value
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            status = "FACE DETECTED"
            color = (0, 255, 0)
        else:
            status = "NO FACE"
            color = (0, 0, 255)

        # Show status
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show detection rate
        rate = (detected_count / frame_count * 100) if frame_count > 0 else 0
        cv2.putText(frame, f"Detection rate: {rate:.1f}%", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Detection Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"Test complete:")
    print(f"  Total frames: {frame_count}")
    print(f"  Faces detected: {detected_count}")
    print(f"  Detection rate: {detected_count / frame_count * 100:.1f}%" if frame_count > 0 else "N/A")
    print("=" * 60)
