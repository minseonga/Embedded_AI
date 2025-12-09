# face_landmark/pipeline.py
# PyTorch face detector (UltraFace) + lightweight MAR estimation.
# Fully removes OpenCV/ONNXRuntime inference and keeps a single torch path.

import os
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnx
from onnx import helper as onnx_helper
import torch
from onnx2pytorch import ConvertModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "assets" / "models"

# Model URL (lightweight, reliable)
MODEL_URLS = {
    "ultraface_onnx": "https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/models/onnx/version-RFB-320_simplified.onnx",
}


def download_file(url: str, filepath: str) -> str:
    """Download file if not exists."""
    if os.path.exists(filepath):
        return filepath

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"[FaceLandmark] Downloading {os.path.basename(filepath)}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"[FaceLandmark] Downloaded to {filepath}")
        return filepath
    except Exception as e:
        print(f"[FaceLandmark] Download failed: {e}")
        raise


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float = 0.35, top_k: int = 200) -> list:
    """Standard NMS for UltraFace."""
    order = scores.argsort()[::-1][:top_k]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / ((
            (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) +
            (boxes[order[1:], 2] - boxes[order[1:], 0]) *
            (boxes[order[1:], 3] - boxes[order[1:], 1]) - inter
        ) + 1e-6)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def _generate_ultraface_priors(input_w: int, input_h: int) -> np.ndarray:
    """Generate priors for UltraFace RFB-320."""
    min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    steps = [8, 16, 32, 64]
    feature_maps = [[int(np.ceil(input_h / step)), int(np.ceil(input_w / step))] for step in steps]

    priors = []
    for k, f in enumerate(feature_maps):
        for i in range(f[0]):
            for j in range(f[1]):
                for min_size in min_sizes[k]:
                    s_kx = min_size / input_w
                    s_ky = min_size / input_h
                    cx = (j + 0.5) * steps[k] / input_w
                    cy = (i + 0.5) * steps[k] / input_h
                    priors.append([cx, cy, s_kx, s_ky])
    return np.array(priors, dtype=np.float32)


class TorchOnnxWrapper:
    """Wrap ONNX graph as torch module."""

    def __init__(self, onnx_path: str, device: torch.device, precision: str = "fp16"):
        model = onnx.load(onnx_path)
        model = self._patch_conv_kernel_size(model)

        self.model = None
        last_err = None
        for exp_flag in (True, False):
            try:
                self.model = ConvertModel(model, experimental=exp_flag)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        if self.model is None:
            raise RuntimeError(f"Failed to convert ONNX to PyTorch: {last_err}") from last_err

        self.model.eval()
        if precision == "fp16":
            self.model.half()
        self.model.to(device)
        self.device = device
        self.precision = precision

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            return self.model(x)

    @staticmethod
    def _patch_conv_kernel_size(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Ensure Conv nodes expose kernel_size attr expected by onnx2pytorch."""
        init_map = {init.name: init for init in onnx_model.graph.initializer}
        for node in onnx_model.graph.node:
            if node.op_type != "Conv":
                continue
            attr_names = {attr.name for attr in node.attribute}
            if "kernel_size" in attr_names:
                continue
            kernel_shape_attr = next((a for a in node.attribute if a.name == "kernel_shape"), None)
            kernel_shape = list(kernel_shape_attr.ints) if kernel_shape_attr is not None else None

            if kernel_shape is None and len(node.input) > 1:
                weight_name = node.input[1]
                weight_init = init_map.get(weight_name)
                if weight_init and len(weight_init.dims) >= 4:
                    kernel_shape = [int(weight_init.dims[2]), int(weight_init.dims[3])]
                    node.attribute.append(onnx_helper.make_attribute("kernel_shape", kernel_shape))

            if kernel_shape is not None:
                node.attribute.append(onnx_helper.make_attribute("kernel_size", kernel_shape))
        return onnx_model


class UltraFaceDetector:
    """UltraFace RFB-320 face detector using PyTorch runtime."""

    def __init__(self, onnx_path: str, device: torch.device, precision: str = "fp16"):
        self.wrapper = TorchOnnxWrapper(onnx_path, device, precision=precision)
        self.device = device
        self.precision = precision
        self.input_w = 320
        self.input_h = 240
        self.priors = _generate_ultraface_priors(self.input_w, self.input_h)
        self.variances = [0.1, 0.2]
        self.conf_threshold = 0.6
        self.nms_threshold = 0.35
        self.top_k = 200

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = img.astype(np.float32)
        img = (img - 127.0) / 128.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        if self.precision == "fp16":
            img = img.half()
        return img.to(self.device)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Return array of [x, y, w, h, score]."""
        inp = self._preprocess(frame)
        outputs = self.wrapper(inp)

        if isinstance(outputs, dict):
            loc = outputs.get("loc") or outputs.get("output_0")
            conf = outputs.get("conf") or outputs.get("output_1")
        elif isinstance(outputs, (list, tuple)):
            loc, conf = outputs[0], outputs[1]
        else:
            raise RuntimeError("Unexpected UltraFace output type")

        loc = loc.detach().cpu().numpy().reshape(-1, 4)
        conf = conf.detach().cpu().numpy().reshape(-1, 2)

        boxes = np.concatenate((
            self.priors[:, :2] + loc[:, :2] * self.variances[0] * self.priors[:, 2:],
            self.priors[:, 2:] * np.exp(loc[:, 2:] * self.variances[1])
        ), axis=1)

        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]

        scores = _softmax(conf)[:, 1]

        mask = scores > self.conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return np.array([])

        # Scale to original frame size
        h, w = frame.shape[:2]
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h

        keep = _nms(boxes, scores, thresh=self.nms_threshold, top_k=self.top_k)
        boxes = boxes[keep]
        scores = scores[keep]

        faces = []
        for b, s in zip(boxes, scores):
            x1, y1, x2, y2 = b
            faces.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(s)])
        return np.array(faces) if faces else np.array([])


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
            ultraface_path = str(MODEL_DIR / "ultraface_rfb_320_simplified.onnx")
            download_file(MODEL_URLS["ultraface_onnx"], ultraface_path)
            self.detector = UltraFaceDetector(ultraface_path, self.device, precision=self.precision)
            print("[FaceLandmark] UltraFace detector loaded (PyTorch)")
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
        det_type = "UltraFace (PyTorch)" if self.detector else "None"
        print(f"[FaceLandmark] Detector: {det_type}")
        print(f"[FaceLandmark] MAR: Ratio-based estimation")


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
