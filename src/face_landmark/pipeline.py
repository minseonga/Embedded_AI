# face_landmark/pipeline.py
# GPU-accelerated Face Landmark Detection using ONNX Runtime
# Replaces MediaPipe Face Mesh for better Jetson Nano performance
# Supports FP16/INT8 quantization like hand_tracking pipeline

import os
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "assets" / "models"

# Model URLs (lightweight models for embedded devices)
MODEL_URLS = {
    "face_detector": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_landmark_68": "https://github.com/atksh/onnx-facial-lmk-detector/raw/main/facial_lmk_detector.onnx",
}

# 68-landmark mouth indices (for smile detection)
MOUTH_LANDMARKS_68 = {
    "left_corner": 48,
    "right_corner": 54,
    "upper_lip": 62,
    "lower_lip": 66,
}


def _select_providers():
    """Pick the fastest available ONNX Runtime providers (TensorRT > CUDA > CPU)."""
    preferred = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    available = set(ort.get_available_providers())
    return [p for p in preferred if p in available]


def download_model(name: str, url: str) -> str:
    """Download model if not exists."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    filename = f"{name}.onnx"
    filepath = MODEL_DIR / filename

    if filepath.exists():
        return str(filepath)

    print(f"[FaceLandmark] Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"[FaceLandmark] Downloaded to {filepath}")
    except Exception as e:
        print(f"[FaceLandmark] Download failed: {e}")
        raise

    return str(filepath)


def convert_to_fp16(src_path: str, dst_path: str) -> str:
    """Convert ONNX model to FP16."""
    try:
        from onnxconverter_common import float16
    except ImportError:
        print("[FaceLandmark] onnxconverter_common not installed, skipping FP16 conversion")
        return src_path

    if os.path.exists(dst_path):
        return dst_path

    print(f"[FaceLandmark] Converting to FP16: {dst_path}")
    model = onnx.load(src_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, dst_path)
    return dst_path


def convert_to_int8(src_path: str, dst_path: str) -> str:
    """Convert ONNX model to INT8 (dynamic quantization)."""
    if os.path.exists(dst_path):
        return dst_path

    print(f"[FaceLandmark] Converting to INT8: {dst_path}")
    quantize_dynamic(src_path, dst_path, weight_type=QuantType.QUInt8)
    return dst_path


def get_model_path(base_name: str, precision: str) -> str:
    """Get model path with precision suffix."""
    if precision == 'fp32':
        return str(MODEL_DIR / f"{base_name}.onnx")
    elif precision == 'fp16':
        return str(MODEL_DIR / f"{base_name}_fp16.onnx")
    elif precision == 'int8':
        return str(MODEL_DIR / f"{base_name}_int8.onnx")
    return str(MODEL_DIR / f"{base_name}.onnx")


class YuNetFaceDetector:
    """YuNet face detector (OpenCV Zoo, very fast)."""

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (320, 320)):
        self.input_size = input_size
        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            input_size,
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000
        )

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect faces. Returns array of [x, y, w, h, ...landmarks, score]."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else np.array([])


class ONNXFaceLandmark:
    """ONNX-based 68-point face landmark detector."""

    def __init__(self, model_path: str, providers: List[str] = None):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path, sess_options,
            providers=providers or _select_providers()
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        # Typically [1, 3, 192, 192] or similar
        self.input_size = (self.input_shape[3], self.input_shape[2])

    def predict(self, face_img: np.ndarray) -> np.ndarray:
        """Predict 68 landmarks from cropped face image.

        Args:
            face_img: BGR face crop

        Returns:
            landmarks: (68, 2) array of (x, y) normalized to [0, 1]
        """
        # Preprocess
        img = cv2.resize(face_img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]  # NCHW

        # Inference
        outputs = self.session.run(None, {self.input_name: img})

        # Parse output - shape varies by model
        landmarks = outputs[0]
        if landmarks.ndim == 3:
            landmarks = landmarks[0]
        landmarks = landmarks.reshape(-1, 2)

        return landmarks


class SimpleFaceLandmark:
    """Simpler approach using OpenCV's built-in face landmark detection.
    Falls back to this if ONNX model download fails."""

    def __init__(self):
        # Use dlib-style facial landmark indices mapping
        # OpenCV's LBF model outputs 68 landmarks
        self.model_path = None
        self.facemark = None

        try:
            # Try to create LBF facemark
            self.facemark = cv2.face.createFacemarkLBF()
            # Try common model paths
            model_paths = [
                "/usr/share/opencv4/lbfmodel.yaml",
                str(MODEL_DIR / "lbfmodel.yaml"),
            ]
            for path in model_paths:
                if os.path.exists(path):
                    self.facemark.loadModel(path)
                    self.model_path = path
                    break
        except Exception:
            self.facemark = None

    @property
    def is_available(self):
        return self.facemark is not None and self.model_path is not None


class FaceLandmarkPipeline:
    """GPU-accelerated face landmark pipeline for Jetson Nano."""

    def __init__(
        self,
        precision: str = 'fp16',
        providers: List[str] = None,
        use_yunet: bool = True,
    ):
        self.precision = precision
        self.providers = providers or _select_providers()

        # Download/load models
        print("[FaceLandmark] Initializing GPU-accelerated face landmark pipeline...")
        print(f"[FaceLandmark] Available providers: {self.providers}")

        # Face detector (YuNet - very fast, uses OpenCV DNN which can use CUDA)
        if use_yunet:
            try:
                det_path = download_model("face_detection_yunet", MODEL_URLS["face_detector"])
                self.detector = YuNetFaceDetector(det_path)
                print("[FaceLandmark] YuNet face detector loaded")
            except Exception as e:
                print(f"[FaceLandmark] YuNet load failed: {e}, using Haar cascade fallback")
                self.detector = None
        else:
            self.detector = None

        # Haar cascade fallback
        if self.detector is None:
            # Handle both pip opencv and apt-get python3-opencv
            cascade_paths = [
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",  # apt-get opencv4
                "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",   # apt-get opencv3
                "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml",   # older versions
            ]
            # Try cv2.data first (pip install opencv-python)
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_paths.insert(0, cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            cascade_path = None
            for path in cascade_paths:
                if os.path.exists(path):
                    cascade_path = path
                    break

            if cascade_path:
                self.haar_cascade = cv2.CascadeClassifier(cascade_path)
                print(f"[FaceLandmark] Using Haar cascade: {cascade_path}")
            else:
                print("[FaceLandmark] Warning: No Haar cascade found, face detection may fail")
                self.haar_cascade = cv2.CascadeClassifier()
        else:
            self.haar_cascade = None

        # Face landmark model with precision conversion
        try:
            base_path = download_model("face_landmark_68", MODEL_URLS["face_landmark_68"])
            lm_path = self._prepare_model(base_path, "face_landmark_68", precision)
            self.landmark_model = ONNXFaceLandmark(lm_path, providers=self.providers)
            self.lm_size = os.path.getsize(lm_path) / (1024 * 1024)
            print(f"[FaceLandmark] ONNX landmark model loaded ({precision}, {self.lm_size:.2f}MB)")
        except Exception as e:
            print(f"[FaceLandmark] ONNX landmark load failed: {e}")
            self.landmark_model = None
            self.lm_size = 0

        # Stats
        self._det_time = 0
        self._lm_time = 0

    def _prepare_model(self, base_path: str, base_name: str, precision: str) -> str:
        """Prepare model with specified precision."""
        if precision == 'fp32':
            return base_path
        elif precision == 'fp16':
            dst_path = get_model_path(base_name, 'fp16')
            return convert_to_fp16(base_path, dst_path)
        elif precision == 'int8':
            dst_path = get_model_path(base_name, 'int8')
            return convert_to_int8(base_path, dst_path)
        return base_path

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
        """Process frame and return face landmarks.

        Args:
            frame: BGR image

        Returns:
            landmarks: (68, 2) array in image coordinates, or None
            mar: mouth aspect ratio for smile detection
            face_box: [x, y, w, h] of detected face, or None
        """
        h, w = frame.shape[:2]

        # Detect faces
        if self.detector is not None:
            faces = self.detector.detect(frame)
        elif self.haar_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.haar_cascade.detectMultiScale(gray, 1.1, 4)
            if len(rects) > 0:
                # Convert to same format as YuNet [x, y, w, h, ...]
                faces = np.array([[x, y, w, h, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]
                                  for (x, y, w, h) in rects])
            else:
                faces = np.array([])
        else:
            return None, 0.0, None

        if len(faces) == 0:
            return None, 0.0, None

        # Use largest face
        face = faces[0]
        x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])

        # Expand face box slightly for better landmark detection
        pad = int(max(fw, fh) * 0.1)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + fw + pad)
        y2 = min(h, y + fh + pad)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None, 0.0, None

        face_box = np.array([x, y, fw, fh])

        # Get landmarks
        if self.landmark_model is not None:
            landmarks_norm = self.landmark_model.predict(face_crop)

            # Denormalize to image coordinates
            crop_h, crop_w = face_crop.shape[:2]
            landmarks = landmarks_norm.copy()
            landmarks[:, 0] = landmarks_norm[:, 0] * crop_w + x1
            landmarks[:, 1] = landmarks_norm[:, 1] * crop_h + y1

            # Calculate mouth aspect ratio
            mar = self._calculate_mar(landmarks)

            return landmarks, mar, face_box

        return None, 0.0, face_box

    def _calculate_mar(self, landmarks: np.ndarray) -> float:
        """Calculate mouth aspect ratio from 68-point landmarks."""
        if len(landmarks) < 68:
            return 0.0

        # Mouth corners and lips
        left_corner = landmarks[MOUTH_LANDMARKS_68["left_corner"]]
        right_corner = landmarks[MOUTH_LANDMARKS_68["right_corner"]]
        upper_lip = landmarks[MOUTH_LANDMARKS_68["upper_lip"]]
        lower_lip = landmarks[MOUTH_LANDMARKS_68["lower_lip"]]

        # Calculate distances
        mouth_width = np.linalg.norm(right_corner - left_corner)
        mouth_height = np.linalg.norm(lower_lip - upper_lip)

        if mouth_width <= 0:
            return 0.0

        return mouth_height / mouth_width

    def get_mouth_center(self, landmarks: np.ndarray) -> Optional[np.ndarray]:
        """Get mouth center from landmarks."""
        if landmarks is None or len(landmarks) < 68:
            return None

        upper_lip = landmarks[MOUTH_LANDMARKS_68["upper_lip"]]
        lower_lip = landmarks[MOUTH_LANDMARKS_68["lower_lip"]]

        return (upper_lip + lower_lip) / 2

    def print_stats(self):
        """Print pipeline info."""
        print(f"[FaceLandmark] Providers: {self.providers}")
        print(f"[FaceLandmark] Precision: {self.precision}")
        print(f"[FaceLandmark] Detector: {'YuNet' if self.detector else 'Haar Cascade'}")
        print(f"[FaceLandmark] Landmark: {'ONNX 68-point' if self.landmark_model else 'None'} ({self.lm_size:.2f}MB)")


def draw_face_landmarks(frame: np.ndarray, landmarks: np.ndarray, color=(0, 255, 0)):
    """Draw face landmarks on frame."""
    if landmarks is None:
        return

    for i, (x, y) in enumerate(landmarks):
        cv2.circle(frame, (int(x), int(y)), 1, color, -1)

    # Draw mouth outline
    mouth_indices = list(range(48, 68))
    for i in range(len(mouth_indices) - 1):
        pt1 = landmarks[mouth_indices[i]].astype(int)
        pt2 = landmarks[mouth_indices[i + 1]].astype(int)
        cv2.line(frame, tuple(pt1), tuple(pt2), (0, 200, 255), 1)


# Test
if __name__ == "__main__":
    pipeline = FaceLandmarkPipeline()
    pipeline.print_stats()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

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

        if landmarks is not None:
            draw_face_landmarks(frame, landmarks)

        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.putText(frame, f"FPS: {avg_fps:.1f} MAR: {mar:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        smile = "SMILING" if mar > 0.3 else ""
        cv2.putText(frame, smile, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Face Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
