# hand_tracking.py
# PyTorch-only hand detection + landmarks (no onnxruntime).
# We convert the existing ONNX assets to PyTorch with onnx2pytorch,
# keep the same gesture API, and run everything on CUDA if available.

import os
import time
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
import onnx
from onnx import helper as onnx_helper
import torch
from onnx2pytorch import ConvertModel

# Paths
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "assets" / "models"


# ============================================================
# Helpers
# ============================================================

def _closest_prune_ratio(ratio: float) -> int:
    """Snap requested ratio to the nearest cached artifact."""
    choices = [0, 10, 20, 30, 50]
    pct = int(round(ratio * 100))
    return min(choices, key=lambda x: abs(x - pct))


def _resolve_model_path(prefix: str, prune_ratio: float, precision: str) -> Tuple[Path, int]:
    """Resolve the best matching ONNX file for detector/landmark."""
    prune = _closest_prune_ratio(prune_ratio)
    prune_tag = f"_pruned{prune}" if prune > 0 else ""

    # fp16 artifacts live in MODEL_DIR, fp32/int8 in MODEL_DIR/onnx_models
    search_roots: List[Path] = []
    if precision == "fp16":
        search_roots = [MODEL_DIR]
        suffix = "_fp16"
    elif precision == "int8":
        search_roots = [MODEL_DIR / "onnx_models", MODEL_DIR]
        suffix = "_int8"
    else:
        search_roots = [MODEL_DIR / "onnx_models", MODEL_DIR]
        suffix = ""

    name = f"{prefix}{prune_tag}{suffix}.onnx"
    for root in search_roots:
        candidate = root / name
        if candidate.exists():
            return candidate, prune

    tried = ", ".join(str((root / name).resolve()) for root in search_roots)
    raise FileNotFoundError(f"Could not find {name} (searched: {tried})")


def resize_pad(img: np.ndarray, target_size: int = 256) -> Tuple[np.ndarray, Tuple[int, int], float]:
    """Resize to square with padding while keeping aspect ratio."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    pad_y = target_size - new_h
    pad_x = target_size - new_w
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, (top, left), scale


def generate_palm_anchors() -> np.ndarray:
    """Generate MediaPipe Palm anchors (2944 anchors for 256x256 input)."""
    strides = [8, 16, 32, 32, 32]  # matches palm detection config
    min_scale = 0.1484375
    max_scale = 0.75
    aspect_ratios = [1.0]
    anchor_offset = 0.5
    input_size = 256

    num_layers = len(strides)
    feature_map_sizes = [int(np.ceil(input_size / s)) for s in strides]
    scales = [min_scale + (max_scale - min_scale) * i / max(1, num_layers - 1) for i in range(num_layers)]
    scales.append(scales[-1])  # for interpolation

    anchors = []
    for layer_id, stride in enumerate(strides):
        fm_size = feature_map_sizes[layer_id]
        scale = scales[layer_id]
        next_scale = scales[layer_id + 1]
        for y in range(fm_size):
            for x in range(fm_size):
                x_center = (x + anchor_offset) / fm_size
                y_center = (y + anchor_offset) / fm_size

                for ratio in aspect_ratios:
                    ratio_sqrt = np.sqrt(ratio)
                    w = scale * ratio_sqrt
                    h = scale / ratio_sqrt
                    anchors.append([x_center, y_center, w, h])

                scale_next = np.sqrt(scale * next_scale)
                anchors.append([x_center, y_center, scale_next, scale_next])

    return np.array(anchors, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    y1 = np.maximum(box[0], boxes[:, 0])
    x1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[2], boxes[:, 2])
    x2 = np.minimum(box[3], boxes[:, 2])

    inter = np.maximum(0, y2 - y1) * np.maximum(0, x2 - x1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area1 + area2 - inter + 1e-6)


def _weighted_nms(detections: np.ndarray, iou_thresh: float = 0.3) -> np.ndarray:
    """Weighted NMS as used by MediaPipe palm detection."""
    if len(detections) == 0:
        return detections

    output = []
    remaining = np.argsort(-detections[:, -1])

    while len(remaining) > 0:
        idx = remaining[0]
        det = detections[idx]
        output.append(det)
        if len(remaining) == 1:
            break

        remaining = remaining[1:]
        ious = _compute_iou(det[:4], detections[remaining, :4])
        remaining = remaining[ious <= iou_thresh]

    return np.array(output)


def extract_roi(frame: np.ndarray, detections: np.ndarray, scale_factor: float = 2.6, shift_y: float = -0.5):
    """Extract rotated hand ROIs for the landmark model."""
    rois = []
    affines = []

    for det in detections:
        kp0_x, kp0_y = det[4], det[5]  # wrist
        kp2_x, kp2_y = det[8], det[9]  # middle finger

        angle = np.arctan2(kp0_y - kp2_y, kp0_x - kp2_x) - np.pi / 2
        cx = (det[1] + det[3]) / 2
        cy = (det[0] + det[2]) / 2
        box_size = det[3] - det[1]

        cy += shift_y * box_size
        size = box_size * scale_factor

        cos_a, sin_a = np.cos(angle), np.sin(angle)
        src = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]], dtype=np.float32) * size
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        src = src @ rot.T
        src[:, 0] += cx
        src[:, 1] += cy

        dst = np.array([[0, 0], [255, 0], [255, 255]], dtype=np.float32)
        M = cv2.getAffineTransform(src.astype(np.float32), dst)
        roi = cv2.warpAffine(frame, M, (256, 256))
        rois.append(roi)
        affines.append(cv2.invertAffineTransform(M))

    return rois, affines


def denormalize_landmarks(landmarks: np.ndarray, affines: List[np.ndarray]) -> List[np.ndarray]:
    """Convert normalized landmarks back to image coordinates."""
    result = []
    for lm, M_inv in zip(landmarks, affines):
        pts = lm[:, :2] * 256
        ones = np.ones((21, 1))
        pts_h = np.hstack([pts, ones])
        pts_img = pts_h @ M_inv.T

        lm_out = np.zeros_like(lm)
        lm_out[:, :2] = pts_img
        lm_out[:, 2] = lm[:, 2]
        result.append(lm_out)

    return result


# ============================================================
# Visualization helpers
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray, size: int = 2):
    """Draw hand landmarks."""
    for conn in HAND_CONNECTIONS:
        pt1 = tuple(landmarks[conn[0], :2].astype(int))
        pt2 = tuple(landmarks[conn[1], :2].astype(int))
        cv2.line(frame, pt1, pt2, (0, 255, 0), size)

    for pt in landmarks:
        cv2.circle(frame, tuple(pt[:2].astype(int)), size + 1, (0, 200, 255), -1)


def draw_detections(frame: np.ndarray, detections: np.ndarray, scale: float, pad: Tuple[int, int]):
    """Draw detection boxes."""
    for det in detections:
        y1 = int(det[0] * scale * 256 - pad[0])
        x1 = int(det[1] * scale * 256 - pad[1])
        y2 = int(det[2] * scale * 256 - pad[0])
        x2 = int(det[3] * scale * 256 - pad[1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


# ============================================================
# Model wrappers
# ============================================================

class TorchOnnxWrapper:
    """Wrap an ONNX graph as a PyTorch nn.Module via onnx2pytorch."""

    def __init__(self, onnx_path: Union[str, Path], device: torch.device, precision: str = "fp16"):
        onnx_model = onnx.load(str(onnx_path))
        onnx_model = self._patch_conv_kernel_size(onnx_model)
        onnx_model = self._remove_storage_order(onnx_model)

        # Some ONNX exports miss kernel_size attr; patch + retry with/without experimental flag.
        self.model = None
        last_err = None
        for exp_flag in (True, False):
            try:
                self.model = ConvertModel(onnx_model, experimental=exp_flag)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        if self.model is None:
            raise RuntimeError(f"Failed to convert ONNX to PyTorch: {last_err}") from last_err

        self.model.eval()
        self.device = device
        if precision == "fp16":
            self.model.half()
        self.model.to(device)

    def __call__(self, x: torch.Tensor):
        with torch.no_grad():
            return self.model(x)

    @staticmethod
    def _patch_conv_kernel_size(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Ensure Conv/ConvTranspose nodes expose kernel_size attr expected by onnx2pytorch."""
        init_map = {init.name: init for init in onnx_model.graph.initializer}
        for node in onnx_model.graph.node:
            if node.op_type not in ("Conv", "ConvTranspose"):
                continue
            attr_names = {attr.name for attr in node.attribute}
            if "kernel_size" in attr_names:
                continue
            kernel_shape_attr = next((a for a in node.attribute if a.name == "kernel_shape"), None)
            kernel_shape = list(kernel_shape_attr.ints) if kernel_shape_attr is not None else None

            # If kernel_shape missing, infer from weight initializer
            if kernel_shape is None and len(node.input) > 1:
                weight_name = node.input[1]
                weight_init = init_map.get(weight_name)
                if weight_init and len(weight_init.dims) >= 4:
                    # weight dims: [out_channels, in_channels/groups, kh, kw] (Conv)
                    # or [in_channels, out_channels/groups, kh, kw] (ConvTranspose)
                    kernel_shape = [int(weight_init.dims[2]), int(weight_init.dims[3])]
                    node.attribute.append(onnx_helper.make_attribute("kernel_shape", kernel_shape))

            if kernel_shape is not None:
                # node.attribute.append(onnx_helper.make_attribute("kernel_size", kernel_shape))
                pass
        return onnx_model

    @staticmethod
    def _remove_storage_order(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
        """Remove storage_order attribute from all nodes (not supported by onnx2pytorch)."""
        for node in onnx_model.graph.node:
            for i, attr in enumerate(node.attribute):
                if attr.name == "storage_order":
                    node.attribute.pop(i)
                    break
        return onnx_model


class TorchPalmDetector:
    """Palm detector using PyTorch (converted from ONNX)."""

    def __init__(self, onnx_path: Union[str, Path], device: torch.device, precision: str = "fp16"):
        self.wrapper = TorchOnnxWrapper(onnx_path, device, precision)
        self.device = device
        self.precision = precision
        self.anchors = generate_palm_anchors()
        self.x_scale = 256.0
        self.y_scale = 256.0
        self.w_scale = 256.0
        self.h_scale = 256.0
        self.score_thresh = 0.5
        self.nms_thresh = 0.3

    def predict(self, img: np.ndarray) -> np.ndarray:
        """Run detection on a 256x256 RGB image."""
        inp = img.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)  # NCHW
        if self.precision == "fp16":
            inp = inp.half()
        inp = inp.to(self.device)

        outputs = self.wrapper(inp)
        if isinstance(outputs, dict):
            raw_boxes = outputs.get("boxes") or outputs.get("output_0")
            raw_scores = outputs.get("scores") or outputs.get("output_1")
        elif isinstance(outputs, (list, tuple)):
            raw_boxes, raw_scores = outputs[0], outputs[1]
        else:
            raise RuntimeError("Unexpected detector output type")

        raw_boxes = raw_boxes.detach().cpu().numpy()
        raw_scores = raw_scores.detach().cpu().numpy()

        if raw_boxes.ndim == 4:
            raw_boxes = raw_boxes[0]
        if raw_scores.ndim == 4:
            raw_scores = raw_scores[0]

        detections = self._decode(raw_boxes, raw_scores)
        return detections

    def _decode(self, raw_boxes: np.ndarray, raw_scores: np.ndarray) -> np.ndarray:
        scores = _sigmoid(raw_scores[:, 0])
        mask = scores > self.score_thresh
        if not mask.any():
            return np.zeros((0, 19), dtype=np.float32)

        scores = scores[mask]
        boxes = raw_boxes[mask]
        anchors = self.anchors[mask]

        cx = boxes[:, 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        cy = boxes[:, 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
        w = boxes[:, 2] / self.w_scale * anchors[:, 2]
        h = boxes[:, 3] / self.h_scale * anchors[:, 3]

        y1 = cy - h / 2
        x1 = cx - w / 2
        y2 = cy + h / 2
        x2 = cx + w / 2

        keypoints = []
        for k in range(7):
            kx = boxes[:, 4 + k * 2] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            ky = boxes[:, 4 + k * 2 + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            keypoints.extend([kx, ky])
        keypoints = np.stack(keypoints, axis=1)

        detections = np.concatenate(
            [y1[:, None], x1[:, None], y2[:, None], x2[:, None], keypoints, scores[:, None]],
            axis=1,
        )
        return _weighted_nms(detections, iou_thresh=self.nms_thresh)


class TorchHandLandmark:
    """Hand landmark model using PyTorch (converted from ONNX)."""

    def __init__(self, onnx_path: Union[str, Path], device: torch.device, precision: str = "fp16"):
        self.wrapper = TorchOnnxWrapper(onnx_path, device, precision)
        self.device = device
        self.precision = precision

    def predict(self, img: np.ndarray):
        """Run landmark detection on 256x256 RGB image batch."""
        if len(img) == 0:
            return np.zeros((0,)), np.zeros((0,)), np.zeros((0, 21, 3))

        inp = img.astype(np.float32) / 255.0
        if inp.ndim == 3:
            inp = inp[None]
        inp = torch.from_numpy(inp).permute(0, 3, 1, 2)  # NCHW
        if self.precision == "fp16":
            inp = inp.half()
        inp = inp.to(self.device)

        outputs = self.wrapper(inp)
        if isinstance(outputs, dict):
            flags = outputs.get("flag") or outputs.get("output_0")
            handed = outputs.get("handedness") or outputs.get("output_1")
            landmarks = outputs.get("landmarks") or outputs.get("output_2")
        elif isinstance(outputs, (list, tuple)):
            if len(outputs) == 3:
                flags, handed, landmarks = outputs
            elif len(outputs) == 2:
                flags, landmarks = outputs
                handed = torch.zeros_like(flags)
            else:
                flags = outputs[0]
                handed = torch.zeros_like(flags)
                landmarks = torch.zeros((len(flags), 21, 3), device=flags.device)
        else:
            raise RuntimeError("Unexpected landmark output type")

        flags = flags.detach().cpu().numpy().flatten()
        handed = handed.detach().cpu().numpy().flatten()
        landmarks = landmarks.detach().cpu().numpy().reshape(-1, 21, 3).astype(np.float32)
        return flags, handed, landmarks


# ============================================================
# Pipeline
# ============================================================

class HandTrackingPipeline:
    """Fast hand tracking with PyTorch (no onnxruntime)."""

    def __init__(
        self,
        use_quantized: bool = False,
        prune_ratio: float = 0.0,
        precision: str = "fp16",
        model_tag: str = None,
        build_only: bool = False,
        prune_mode: str = "magnitude",
    ):
        del model_tag, prune_mode, use_quantized  # unused; kept for CLI compatibility

        precision = precision.lower()
        if precision not in ("fp32", "fp16", "int8"):
            raise ValueError("precision must be one of: fp32, fp16, int8")
        if precision == "int8":
            print("[HandTracking] int8 is not supported in PyTorch path; falling back to fp16.")
            precision = "fp16"

        self.precision = precision
        self.prune_ratio = prune_ratio
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[HandTracking] Device: {self.device} | Precision: {self.precision}")

        det_path, det_prune = _resolve_model_path("palm_detector", prune_ratio, precision)
        lm_path, lm_prune = _resolve_model_path("hand_landmark", prune_ratio, precision)
        self.prune_ratio = max(det_prune, lm_prune) / 100.0

        if build_only:
            self.detector = None
            self.landmark = None
        else:
            self.detector = TorchPalmDetector(det_path, self.device, precision=self.precision)
            self.landmark = TorchHandLandmark(lm_path, self.device, precision=self.precision)

            if self.prune_ratio > 0:
                print(f"[HandTracking] Applying structured pruning (ratio={self.prune_ratio})...")
                self._prune_model(self.detector.wrapper.model, self.prune_ratio)
                self._prune_model(self.landmark.wrapper.model, self.prune_ratio)

        self.det_size = det_path.stat().st_size / (1024 * 1024)
        self.lm_size = lm_path.stat().st_size / (1024 * 1024)

    def _prune_model(self, model: torch.nn.Module, ratio: float):
        """Apply structured pruning (L1 norm) to Conv2d layers."""
        import torch.nn.utils.prune as prune
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # Prune filters (dim=0) based on L1 norm
                prune.ln_structured(module, name="weight", amount=ratio, n=1, dim=0)
                # Make pruning permanent (remove mask, modify weight directly)
                prune.remove(module, "weight")

    def process_frame(self, frame: np.ndarray):
        """Process a single BGR frame. Returns (landmarks, detections, scale, pad)."""
        if self.detector is None or self.landmark is None:
            return [], [], 1.0, (0, 0)

        frame_rgb = frame[:, :, ::-1].copy()
        img256, pad, scale = resize_pad(frame_rgb)

        detections = self.detector.predict(img256)
        if len(detections) == 0:
            return [], [], scale, pad

        det_img = detections.copy()
        det_img[:, 0] = detections[:, 0] * scale * 256 - pad[0]
        det_img[:, 1] = detections[:, 1] * scale * 256 - pad[1]
        det_img[:, 2] = detections[:, 2] * scale * 256 - pad[0]
        det_img[:, 3] = detections[:, 3] * scale * 256 - pad[1]
        det_img[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]  # x coords
        det_img[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]  # y coords

        rois, affines = extract_roi(frame_rgb, det_img)
        if len(rois) == 0:
            return [], detections, scale, pad

        roi_batch = np.stack(rois, axis=0)
        flags, _, landmarks = self.landmark.predict(roi_batch)

        landmarks_img = denormalize_landmarks(landmarks, affines)
        results = []
        for flag, lm in zip(flags, landmarks_img):
            if flag > 0.5:
                results.append(lm)

        return results, detections, scale, pad

    def run_webcam(self):
        """Run a simple webcam demo."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("Press ESC to quit, 'i' for info")
        fps_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame[:, ::-1].copy()  # Mirror
            start = time.time()
            landmarks, detections, scale, pad = self.process_frame(frame)
            elapsed = time.time() - start

            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)

            for lm in landmarks:
                draw_landmarks(frame, lm)
            if len(detections) > 0:
                draw_detections(frame, detections, scale, pad)

            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Tracking", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('i'):
                self.print_stats()

        cap.release()
        cv2.destroyAllWindows()

    def print_stats(self):
        """Print model stats."""
        print(f"\nDetector: {self.det_size:.2f} MB")
        print(f"Landmark: {self.lm_size:.2f} MB")
        print(f"Precision: {self.precision}")
        print(f"Prune ratio: {self.prune_ratio*100:.0f}%\n")


# ============================================================
# Benchmarking
# ============================================================

def benchmark(pipeline, num_frames=100):
    """Benchmark inference speed."""
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for _ in range(10):
        pipeline.process_frame(dummy_frame)

    times = []
    for _ in range(num_frames):
        start = time.time()
        pipeline.process_frame(dummy_frame)
        times.append(time.time() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "fps": 1000 / (np.mean(times) * 1000),
    }


def run_benchmarks():
    """Run a set of benchmark configs."""
    print("\n" + "=" * 70)
    print("Benchmarking Hand Tracking Pipeline (PyTorch runtime)")
    print("=" * 70)

    configs = [
        {"name": "FP16 (Jetson)", "precision": "fp16", "prune": 0.0},
        {"name": "FP32 baseline", "precision": "fp32", "prune": 0.0},
        {"name": "Pruned 30% FP16", "precision": "fp16", "prune": 0.3},
        {"name": "Pruned 30% FP32", "precision": "fp32", "prune": 0.3},
    ]

    results = []
    for cfg in configs:
        print(f"\nTesting: {cfg['name']}...")
        try:
            pipeline = HandTrackingPipeline(
                prune_ratio=cfg["prune"],
                precision=cfg["precision"],
            )
            stats = benchmark(pipeline, num_frames=50)
            results.append({
                "name": cfg["name"],
                "det_size": pipeline.det_size,
                "lm_size": pipeline.lm_size,
                "total_size": pipeline.det_size + pipeline.lm_size,
                "mean_ms": stats["mean_ms"],
                "fps": stats["fps"],
            })
        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 85)
    print(f"{'Config':<25} {'Det Size':>10} {'LM Size':>10} {'Total':>10} {'Time':>12} {'FPS':>8}")
    print("=" * 85)
    baseline = results[0]["mean_ms"] if results else 1
    for r in results:
        speedup = baseline / r["mean_ms"]
        print(f"{r['name']:<25} {r['det_size']:>8.2f}MB {r['lm_size']:>8.2f}MB "
              f"{r['total_size']:>8.2f}MB {r['mean_ms']:>10.2f}ms {r['fps']:>6.1f} ({speedup:.2f}x)")
    print("=" * 85)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize', action='store_true',
                        help='Legacy flag (ignored, int8 not supported in PyTorch path).')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16',
                        help='Model precision. int8 will auto-fallback to fp16 here.')
    parser.add_argument('--prune', type=float, default=0.0, help='Pruning ratio (0-1).')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')

    args = parser.parse_args()
    precision = args.precision
    if args.quantize:
        precision = 'int8'

    if args.benchmark:
        run_benchmarks()
    else:
        pipeline = HandTrackingPipeline(
            prune_ratio=args.prune,
            precision=precision,
        )
        pipeline.print_stats()
        pipeline.run_webcam()


if __name__ == "__main__":
    main()
