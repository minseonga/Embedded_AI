# face_landmark/pipeline.py
# MediaPipe face detection (works with ANY OpenCV version).
# Optimized for Jetson Nano - no OpenCV DNN/Haar dependencies.

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "assets" / "models"


class MediaPipeFaceDetector:
    """Face detector using MediaPipe (TFLite, works with any OpenCV version).

    Advantages:
    - Works with ALL OpenCV versions (no cv2.data or cv2.dnn dependency)
    - Very fast and accurate
    - No model download needed (included in MediaPipe)
    - Optimized for embedded devices
    """

    def __init__(self, device: torch.device, precision: str = "fp16"):
        self.device = device
        self.precision = precision
        self.frame_count = 0
        self.detection_count = 0

        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection

            # model_selection: 0 for short-range (2m), 1 for full-range (5m)
            # Using model 1 for better detection range
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # Full-range for better detection
                min_detection_confidence=0.3  # Lower threshold for better sensitivity
            )
            print("[FaceLandmark] MediaPipe Face Detection loaded (TFLite, optimized)")
            print("[FaceLandmark] Model: Full-range (5m), Confidence: 0.3")
        except ImportError:
            print("[FaceLandmark] MediaPipe not found, face detection disabled")
            self.face_detection = None
        except Exception as e:
            print(f"[FaceLandmark] MediaPipe initialization error: {e}")
            import traceback
            traceback.print_exc()
            self.face_detection = None

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return array of [x, y, w, h, score]."""
        if self.face_detection is None:
            return np.array([])

        self.frame_count += 1

        try:
            # Validate frame
            if frame is None or frame.size == 0:
                print(f"[FaceLandmark] Invalid frame at count {self.frame_count}")
                return np.array([])

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.face_detection.process(frame_rgb)

            if not results.detections:
                # Log every 100 frames if no detection
                if self.frame_count % 100 == 0:
                    print(f"[FaceLandmark] No face detected in {self.frame_count} frames (detected: {self.detection_count})")
                return np.array([])

            # Convert to [x, y, w, h, score] format
            faces = []
            h, w = frame.shape[:2]

            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                score = detection.score[0]

                # Convert normalized coordinates to pixels
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                # Ensure bbox is within frame
                x = max(0, x)
                y = max(0, y)
                width = min(w - x, width)
                height = min(h - y, height)

                if width > 0 and height > 0:
                    faces.append([x, y, width, height, float(score)])
                    self.detection_count += 1
                    # Log first few detections
                    if self.detection_count <= 3:
                        print(f"[FaceLandmark] Face detected! Box: ({x}, {y}, {width}, {height}), Score: {score:.2f}")

            return np.array(faces) if faces else np.array([])

        except Exception as e:
            print(f"[FaceLandmark] Face detection error at frame {self.frame_count}: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])


class RatioBasedMAR:
    """Simple ratio-based smile detection using face proportions (no face mesh)."""

    def __init__(self):
        self.prev_mar = 0.0
        self.smoothing = 0.3

    def estimate_from_face(self, frame: np.ndarray, face_box: np.ndarray) -> Tuple[float, np.ndarray]:
        x, y, w, h = face_box
        mouth_y = y + int(h * 0.65)
        mouth_x = x + w // 2
        mouth_width = int(w * 0.5)
        mouth_height = int(h * 0.15)

        mouth_y1 = max(0, mouth_y - mouth_height // 2)
        mouth_y2 = min(frame.shape[0], mouth_y + mouth_height // 2)
        mouth_x1 = max(0, mouth_x - mouth_width // 2)
        mouth_x2 = min(frame.shape[1], mouth_x + mouth_width // 2)

        mouth_roi = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        if mouth_roi.size == 0:
            return self.prev_mar, np.array([mouth_x, mouth_y])

        gray = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY) if len(mouth_roi.shape) == 3 else mouth_roi
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        variance = np.var(gray) / 255.0

        mar = edge_density * 0.5 + variance * 0.3
        mar = max(0.0, min(1.0, mar))
        mar = self.smoothing * mar + (1 - self.smoothing) * self.prev_mar
        self.prev_mar = mar

        return mar, np.array([mouth_x, mouth_y])


class FaceLandmarkPipeline:
    """Face detection pipeline for Jetson Nano + MAR estimation.

    Uses MediaPipe Face Detection (works with ANY OpenCV version).
    """

    def __init__(
        self,
        precision: str = "fp16",
        use_cuda: bool = True,
        use_dnn: bool = False,  # Ignored, kept for compatibility
    ):
        del use_dnn  # Ignored parameter

        self.precision = precision
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        print("[FaceLandmark] Initializing face detection pipeline...")
        print(f"[FaceLandmark] Device: {self.device}")

        self.detector = None
        try:
            self.detector = MediaPipeFaceDetector(self.device, precision=self.precision)
            print("[FaceLandmark] MediaPipe Face Detection ready")
        except Exception as e:
            print(f"[FaceLandmark] Face detector failed: {e}")
            self.detector = None

        self.mar_estimator = RatioBasedMAR()

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
        """Process frame and return (landmarks, mar, face_box)."""
        if self.detector is None:
            return None, 0.0, None

        faces = self.detector.detect(frame)
        if len(faces) == 0:
            return None, 0.0, None

        if len(faces) > 1:
            areas = faces[:, 2] * faces[:, 3]
            face = faces[np.argmax(areas)]
        else:
            face = faces[0]

        face_box = np.array([int(face[0]), int(face[1]), int(face[2]), int(face[3])])
        mar, _ = self.mar_estimator.estimate_from_face(frame, face_box)
        return None, mar, face_box

    def get_mouth_center(self, landmarks: np.ndarray = None, face_box: np.ndarray = None) -> Optional[np.ndarray]:
        """Get mouth center estimate from face box."""
        del landmarks  # unused
        if face_box is not None:
            x, y, w, h = face_box
            return np.array([x + w // 2, y + int(h * 0.65)])
        return None

    def print_stats(self):
        print(f"[FaceLandmark] Device: {self.device}")
        print(f"[FaceLandmark] Precision: {self.precision}")
        det_type = "MediaPipe Face Detection (TFLite)" if self.detector else "None"
        print(f"[FaceLandmark] Detector: {det_type}")
        print(f"[FaceLandmark] MAR: Ratio-based estimation")
        print(f"[FaceLandmark] Works with ANY OpenCV version")


def draw_face_box(frame: np.ndarray, face_box: np.ndarray, color=(255, 0, 0)):
    """Draw face bounding box on frame."""
    if face_box is None:
        return
    x, y, w, h = face_box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


if __name__ == "__main__":
    pipeline = FaceLandmarkPipeline()
    pipeline.print_stats()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        raise SystemExit

    import time
    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[:, ::-1].copy()

        start = time.time()
        landmarks, mar, face_box = pipeline.process_frame(frame)
        elapsed = time.time() - start

        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_list.append(fps)
        if len(fps_list) > 30:
            fps_list.pop(0)
        avg_fps = np.mean(fps_list)

        if face_box is not None:
            draw_face_box(frame, face_box)
            mouth_center = pipeline.get_mouth_center(None, face_box)
            if mouth_center is not None:
                cv2.circle(frame, tuple(mouth_center.astype(int)), 5, (0, 255, 255), -1)

        cv2.putText(frame, f"FPS: {avg_fps:.1f} MAR: {mar:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "SMILING" if mar > 0.3 else "", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
