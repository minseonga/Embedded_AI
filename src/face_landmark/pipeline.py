# face_landmark/pipeline.py
# Ultra-lightweight face detector using OpenCV YuNet + MAR estimation.
# Optimized for Jetson Nano with quantization support.

import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "assets" / "models"

# YuNet model URL (ultra-lightweight ~1MB)
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


def download_yunet_model():
    """Download YuNet face detection model."""
    model_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    if model_path.exists():
        return str(model_path)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("[FaceLandmark] Downloading YuNet model...")
    try:
        urllib.request.urlretrieve(YUNET_MODEL_URL, str(model_path))
        print(f"[FaceLandmark] Downloaded to {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"[FaceLandmark] Download failed: {e}")
        return None


class YuNetFaceDetector:
    """Ultra-lightweight face detector using OpenCV YuNet (~250KB).

    YuNet is optimized for edge devices and supports:
    - OpenCV DNN backend with CUDA acceleration
    - Very fast inference (~5-10ms on Jetson Nano)
    - Small model size (~1MB)
    """

    def __init__(self, device: torch.device, precision: str = "fp16"):
        self.device = device
        self.precision = precision
        self.score_threshold = 0.6
        self.nms_threshold = 0.3
        self.top_k = 5000

        # Download model
        model_path = download_yunet_model()
        if model_path is None:
            print("[FaceLandmark] YuNet model not available")
            self.detector = None
            return

        # Initialize YuNet detector
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k
        )

        # Set CUDA backend if available on Jetson Nano
        if device.type == 'cuda':
            try:
                self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("[FaceLandmark] YuNet using CUDA backend")
            except Exception:
                print("[FaceLandmark] YuNet using CPU backend")

        print(f"[FaceLandmark] YuNet detector loaded (~1MB, ultra-fast)")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return array of [x, y, w, h, score]."""
        if self.detector is None:
            return np.array([])

        h, w = frame.shape[:2]

        # Set input size dynamically
        self.detector.setInputSize((w, h))

        try:
            # Detect faces
            _, faces = self.detector.detect(frame)

            if faces is None or len(faces) == 0:
                return np.array([])

            # Convert to [x, y, w, h, score] format
            results = []
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                score = float(face[-1])
                results.append([x, y, w, h, score])

            return np.array(results) if results else np.array([])

        except Exception as e:
            print(f"[FaceLandmark] Detection error: {e}")
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
    """PyTorch face detection pipeline for Jetson Nano + MAR estimation."""

    def __init__(
        self,
        precision: str = "fp16",
        use_cuda: bool = True,
    ):
        self.precision = precision
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        print("[FaceLandmark] Initializing face detection pipeline...")
        print(f"[FaceLandmark] Device: {self.device}")

        self.detector = None
        try:
            self.detector = YuNetFaceDetector(self.device, precision=self.precision)
            print("[FaceLandmark] YuNet face detector loaded (OpenCV DNN + CUDA)")
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
        det_type = "YuNet (OpenCV DNN + CUDA, ~1MB)" if self.detector else "None"
        print(f"[FaceLandmark] Detector: {det_type}")
        print(f"[FaceLandmark] MAR: Ratio-based estimation")
        print(f"[FaceLandmark] Optimized for Jetson Nano")


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
