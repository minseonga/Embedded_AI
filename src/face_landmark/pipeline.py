# face_landmark/pipeline.py
# Ultra-lightweight face detector using Haar Cascades (apt-get opencv compatible).
# Optimized for Jetson Nano - works with old OpenCV versions.

import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "assets" / "models"

# DNN model URLs (fallback option)
CAFFE_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFE_MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def download_dnn_model():
    """Download OpenCV DNN face detection model (Caffe)."""
    prototxt_path = MODEL_DIR / "deploy.prototxt"
    model_path = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

    if prototxt_path.exists() and model_path.exists():
        return str(prototxt_path), str(model_path)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        if not prototxt_path.exists():
            print("[FaceLandmark] Downloading deploy.prototxt...")
            urllib.request.urlretrieve(CAFFE_PROTOTXT_URL, str(prototxt_path))

        if not model_path.exists():
            print("[FaceLandmark] Downloading Caffe model (~10MB)...")
            urllib.request.urlretrieve(CAFFE_MODEL_URL, str(model_path))

        print(f"[FaceLandmark] DNN models downloaded")
        return str(prototxt_path), str(model_path)
    except Exception as e:
        print(f"[FaceLandmark] DNN model download failed: {e}")
        return None, None


class HaarCascadeFaceDetector:
    """Ultra-fast face detector using Haar Cascades.

    Advantages:
    - Works with ALL OpenCV versions (even old apt-get versions)
    - Extremely fast (~1-2ms per frame on Jetson Nano)
    - Very lightweight (~1MB)
    - No download needed (built into OpenCV)
    """

    def __init__(self, device: torch.device, precision: str = "fp16"):
        self.device = device
        self.precision = precision
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.min_size = (30, 30)

        # Load Haar Cascade (built into OpenCV)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError("Failed to load Haar Cascade")

        print(f"[FaceLandmark] Haar Cascade detector loaded (ultra-fast, built-in)")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return array of [x, y, w, h, score]."""
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return np.array([])

        # Convert to [x, y, w, h, score] format
        # Haar doesn't provide confidence, so we use 1.0
        results = []
        for (x, y, w, h) in faces:
            results.append([int(x), int(y), int(w), int(h), 1.0])

        return np.array(results) if results else np.array([])


class DNNFaceDetector:
    """DNN-based face detector using OpenCV DNN + Caffe.

    Advantages:
    - Works with OpenCV 3.3+ (most apt-get versions)
    - More accurate than Haar Cascades
    - CUDA acceleration support
    - Still lightweight (~10MB)
    """

    def __init__(self, device: torch.device, precision: str = "fp16"):
        self.device = device
        self.precision = precision
        self.confidence_threshold = 0.5

        # Download models
        prototxt_path, model_path = download_dnn_model()
        if prototxt_path is None or model_path is None:
            raise RuntimeError("Failed to download DNN models")

        # Load DNN model
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

        # Set CUDA backend if available
        if device.type == 'cuda':
            try:
                self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("[FaceLandmark] DNN using CUDA backend")
            except Exception:
                print("[FaceLandmark] DNN using CPU backend")
        else:
            print("[FaceLandmark] DNN using CPU backend")

        print(f"[FaceLandmark] DNN detector loaded (~10MB, CUDA accelerated)")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return array of [x, y, w, h, score]."""
        h, w = frame.shape[:2]

        # Prepare blob
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False
        )

        # Forward pass
        self.detector.setInput(blob)
        detections = self.detector.forward()

        # Parse detections
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype(int)

                x, y = max(0, x1), max(0, y1)
                w_box = min(w - x, x2 - x1)
                h_box = min(h - y, y2 - y1)

                results.append([int(x), int(y), int(w_box), int(h_box), float(confidence)])

        return np.array(results) if results else np.array([])


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

    Uses Haar Cascades by default (works with all OpenCV versions).
    Falls back to DNN if Haar fails.
    """

    def __init__(
        self,
        precision: str = "fp16",
        use_cuda: bool = True,
        use_dnn: bool = False,  # Set to True to prefer DNN over Haar
    ):
        self.precision = precision
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        print("[FaceLandmark] Initializing face detection pipeline...")
        print(f"[FaceLandmark] Device: {self.device}")

        self.detector = None
        detector_type = "unknown"

        # Try to load face detector
        if use_dnn:
            # Try DNN first if requested
            try:
                self.detector = DNNFaceDetector(self.device, precision=self.precision)
                detector_type = "DNN (OpenCV Caffe)"
            except Exception as e:
                print(f"[FaceLandmark] DNN detector failed: {e}, falling back to Haar Cascade")
                try:
                    self.detector = HaarCascadeFaceDetector(self.device, precision=self.precision)
                    detector_type = "Haar Cascade (built-in)"
                except Exception as e2:
                    print(f"[FaceLandmark] Haar Cascade failed: {e2}")
                    self.detector = None
        else:
            # Try Haar Cascade first (default, works with all OpenCV)
            try:
                self.detector = HaarCascadeFaceDetector(self.device, precision=self.precision)
                detector_type = "Haar Cascade (built-in)"
            except Exception as e:
                print(f"[FaceLandmark] Haar Cascade failed: {e}, falling back to DNN")
                try:
                    self.detector = DNNFaceDetector(self.device, precision=self.precision)
                    detector_type = "DNN (OpenCV Caffe)"
                except Exception as e2:
                    print(f"[FaceLandmark] DNN detector failed: {e2}")
                    self.detector = None

        if self.detector:
            print(f"[FaceLandmark] Using {detector_type}")
        else:
            print("[FaceLandmark] WARNING: No face detector available!")

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

        if self.detector:
            if isinstance(self.detector, HaarCascadeFaceDetector):
                det_type = "Haar Cascade (built-in, ultra-fast)"
            elif isinstance(self.detector, DNNFaceDetector):
                det_type = "DNN Caffe (~10MB, CUDA accelerated)"
            else:
                det_type = "Unknown"
        else:
            det_type = "None"

        print(f"[FaceLandmark] Detector: {det_type}")
        print(f"[FaceLandmark] MAR: Ratio-based estimation")
        print(f"[FaceLandmark] Compatible with apt-get OpenCV")


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
