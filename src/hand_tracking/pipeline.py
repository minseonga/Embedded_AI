# hand_tracking.py
# MediaPipe Hand Detection + Landmark
# Real speedup with: Channel Pruning + ONNX Runtime + INT8 Quantization

import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths
ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "assets" / "models"
MPT_DIR = BASE_DIR / "mediapipe_pytorch"

# Add MediaPipePyTorch to path
if str(MPT_DIR) not in sys.path:
    sys.path.insert(0, str(MPT_DIR))

from blazebase import resize_pad, denormalize_detections  # type: ignore
from blazepalm import BlazePalm  # type: ignore
from blazehand_landmark import BlazeHandLandmark  # type: ignore


def _select_providers():
    """Pick the fastest available ONNX Runtime providers (TensorRT > CUDA > CPU)."""
    preferred = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    available = set(ort.get_available_providers())
    return [p for p in preferred if p in available]


# ============================================================
# Channel Pruning - Actually removes channels
# ============================================================

def get_conv_layers(model):
    """Get all Conv2d layers from model."""
    convs = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            convs.append((name, module))
    return convs


def prune_conv_output_channels(conv, keep_mask):
    """Prune output channels of a Conv2d layer."""
    keep_indices = torch.where(keep_mask)[0]

    new_out_channels = len(keep_indices)
    new_weight = conv.weight.data[keep_indices]
    new_bias = conv.bias.data[keep_indices] if conv.bias is not None else None

    new_conv = nn.Conv2d(
        conv.in_channels, new_out_channels, conv.kernel_size,
        conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None
    )
    new_conv.weight.data = new_weight
    if new_bias is not None:
        new_conv.bias.data = new_bias

    return new_conv, keep_indices


def prune_conv_input_channels(conv, keep_indices):
    """Prune input channels of a Conv2d layer (for non-depthwise)."""
    if conv.groups > 1:  # Depthwise conv
        return conv

    new_in_channels = len(keep_indices)
    new_weight = conv.weight.data[:, keep_indices]

    new_conv = nn.Conv2d(
        new_in_channels, conv.out_channels, conv.kernel_size,
        conv.stride, conv.padding, conv.dilation, conv.groups, conv.bias is not None
    )
    new_conv.weight.data = new_weight
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data

    return new_conv


def compute_importance(weight):
    """Compute L1-norm importance per output channel."""
    return weight.abs().sum(dim=(1, 2, 3))


def get_pruning_mask(importance, prune_ratio):
    """Get boolean mask for channels to keep."""
    n = len(importance)
    n_keep = max(1, int(n * (1 - prune_ratio)))
    threshold = torch.topk(importance, n_keep).values.min()
    return importance >= threshold


# ============================================================
# Export to ONNX
# ============================================================

def export_to_onnx(model, output_path, input_shape=(1, 3, 256, 256)):
    """Export PyTorch model to ONNX."""
    model.eval()
    dummy = torch.randn(*input_shape)

    torch.onnx.export(
        model, dummy, output_path,
        input_names=['input'],
        output_names=['output_0', 'output_1'],
        opset_version=13,
        do_constant_folding=True
    )
    return output_path


def quantize_onnx_model(input_path, output_path):
    """Quantize ONNX model to INT8."""
    quantize_dynamic(
        input_path, output_path,
        weight_type=QuantType.QUInt8
    )
    return output_path


# ============================================================
# ONNX Runtime Inference
# ============================================================

class ONNXDetector:
    """ONNX Runtime based palm detector."""

    def __init__(self, onnx_path, anchors_path, providers: List[str] = None):
        # Use all available optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            onnx_path, sess_options,
            providers=providers or _select_providers()
        )
        self.input_name = self.session.get_inputs()[0].name
        self.anchors = np.load(anchors_path).astype(np.float32)

        # Detection params
        self.x_scale = 256.0
        self.y_scale = 256.0
        self.w_scale = 256.0
        self.h_scale = 256.0
        self.score_thresh = 0.5
        self.nms_thresh = 0.3

    def predict(self, img):
        """Run detection on 256x256 RGB image."""
        # Preprocess
        inp = img.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None]  # NCHW

        # Run inference
        outputs = self.session.run(None, {self.input_name: inp})
        raw_boxes, raw_scores = outputs[0], outputs[1]

        # Decode
        detections = self._decode(raw_boxes[0], raw_scores[0])
        return detections

    def _decode(self, raw_boxes, raw_scores):
        """Decode raw outputs to detections."""
        # raw_boxes: (2944, 18), raw_scores: (2944, 1)
        scores = self._sigmoid(raw_scores[:, 0])

        # Filter by score
        mask = scores > self.score_thresh
        if not mask.any():
            return np.zeros((0, 19), dtype=np.float32)

        scores = scores[mask]
        boxes = raw_boxes[mask]
        anchors = self.anchors[mask]

        # Decode boxes
        cx = boxes[:, 0] / self.x_scale * anchors[:, 2] + anchors[:, 0]
        cy = boxes[:, 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
        w = boxes[:, 2] / self.w_scale * anchors[:, 2]
        h = boxes[:, 3] / self.h_scale * anchors[:, 3]

        # ymin, xmin, ymax, xmax
        y1 = cy - h / 2
        x1 = cx - w / 2
        y2 = cy + h / 2
        x2 = cx + w / 2

        # Keypoints
        keypoints = []
        for k in range(7):
            kx = boxes[:, 4 + k*2] / self.x_scale * anchors[:, 2] + anchors[:, 0]
            ky = boxes[:, 4 + k*2 + 1] / self.y_scale * anchors[:, 3] + anchors[:, 1]
            keypoints.extend([kx, ky])

        keypoints = np.stack(keypoints, axis=1)  # (N, 14)

        detections = np.concatenate([
            y1[:, None], x1[:, None], y2[:, None], x2[:, None],
            keypoints,
            scores[:, None]
        ], axis=1)

        # NMS
        detections = self._weighted_nms(detections)
        return detections

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    def _weighted_nms(self, detections):
        """Weighted NMS as in MediaPipe."""
        if len(detections) == 0:
            return detections

        output = []
        remaining = np.argsort(-detections[:, -1])  # Sort by score

        while len(remaining) > 0:
            idx = remaining[0]
            det = detections[idx]
            output.append(det)

            if len(remaining) == 1:
                break

            remaining = remaining[1:]
            ious = self._compute_iou(det[:4], detections[remaining, :4])
            remaining = remaining[ious <= self.nms_thresh]

        return np.array(output)

    def _compute_iou(self, box, boxes):
        """Compute IoU between one box and multiple boxes."""
        y1 = np.maximum(box[0], boxes[:, 0])
        x1 = np.maximum(box[1], boxes[:, 1])
        y2 = np.minimum(box[2], boxes[:, 2])
        x2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, y2 - y1) * np.maximum(0, x2 - x1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return inter / (area1 + area2 - inter + 1e-6)


class ONNXLandmark:
    """ONNX Runtime based hand landmark detector."""

    def __init__(self, onnx_path, providers: List[str] = None):
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            onnx_path, sess_options,
            providers=providers or _select_providers()
        )
        self.input_name = self.session.get_inputs()[0].name
        self.resolution = 256

    def predict(self, img):
        """Run landmark detection on 256x256 RGB image batch."""
        if len(img) == 0:
            return np.zeros((0,)), np.zeros((0,)), np.zeros((0, 21, 3))

        # Preprocess
        inp = img.astype(np.float32) / 255.0
        if inp.ndim == 3:
            inp = inp[None]
        inp = inp.transpose(0, 3, 1, 2)  # NCHW

        # Run inference
        outputs = self.session.run(None, {self.input_name: inp})

        # Parse outputs - may vary by model version
        if len(outputs) == 3:
            flags, handed, landmarks = outputs
        else:
            flags = outputs[0]
            handed = np.zeros_like(flags)
            landmarks = outputs[1] if len(outputs) > 1 else np.zeros((len(flags), 21, 3))

        # Ensure correct shapes
        flags = np.asarray(flags).flatten()
        handed = np.asarray(handed).flatten()
        # ONNX model already outputs landmarks normalized to the ROI size
        landmarks = np.asarray(landmarks).reshape(-1, 21, 3).astype(np.float32)

        return flags, handed, landmarks


# ============================================================
# ROI Extraction (from blazebase)
# ============================================================

def extract_roi(frame, detections, scale_factor=2.6, shift_y=-0.5):
    """Extract hand ROIs from detections."""
    rois = []
    affines = []

    for det in detections:
        # Get keypoints for rotation
        kp0_x, kp0_y = det[4], det[5]  # Wrist
        kp2_x, kp2_y = det[8], det[9]  # Middle finger

        # Compute rotation angle
        angle = np.arctan2(kp0_y - kp2_y, kp0_x - kp2_x) - np.pi/2

        # Compute center and scale from box
        cx = (det[1] + det[3]) / 2
        cy = (det[0] + det[2]) / 2
        box_size = det[3] - det[1]  # x-extent

        # Apply shift and scale
        cy += shift_y * box_size
        size = box_size * scale_factor

        # Create affine transform
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        # Source points (unit square centered at origin)
        src = np.array([
            [-0.5, -0.5],
            [0.5, -0.5],
            [0.5, 0.5]
        ], dtype=np.float32) * size

        # Rotate
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        src = src @ rot.T

        # Translate to center
        src[:, 0] += cx
        src[:, 1] += cy

        # Destination points
        dst = np.array([
            [0, 0],
            [255, 0],
            [255, 255]
        ], dtype=np.float32)

        # Compute affine matrix
        M = cv2.getAffineTransform(src.astype(np.float32), dst)

        # Warp image
        roi = cv2.warpAffine(frame, M, (256, 256))
        rois.append(roi)

        # Store inverse affine for landmark denormalization
        M_inv = cv2.invertAffineTransform(M)
        affines.append(M_inv)

    return rois, affines


def denormalize_landmarks(landmarks, affines):
    """Convert normalized landmarks to image coordinates."""
    result = []
    for lm, M_inv in zip(landmarks, affines):
        pts = lm[:, :2] * 256  # Scale to ROI size
        ones = np.ones((21, 1))
        pts_h = np.hstack([pts, ones])
        pts_img = pts_h @ M_inv.T

        lm_out = np.zeros_like(lm)
        lm_out[:, :2] = pts_img
        lm_out[:, 2] = lm[:, 2]
        result.append(lm_out)

    return result


# ============================================================
# Visualization
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_landmarks(frame, landmarks, size=2):
    """Draw hand landmarks."""
    for conn in HAND_CONNECTIONS:
        pt1 = tuple(landmarks[conn[0], :2].astype(int))
        pt2 = tuple(landmarks[conn[1], :2].astype(int))
        cv2.line(frame, pt1, pt2, (0, 255, 0), size)

    for pt in landmarks:
        cv2.circle(frame, tuple(pt[:2].astype(int)), size+1, (0, 200, 255), -1)


def draw_detections(frame, detections, scale, pad):
    """Draw detection boxes."""
    for det in detections:
        y1 = int(det[0] * scale * 256 - pad[0])
        x1 = int(det[1] * scale * 256 - pad[1])
        y2 = int(det[2] * scale * 256 - pad[0])
        x2 = int(det[3] * scale * 256 - pad[1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


# ============================================================
# Pipeline
# ============================================================

class HandTrackingPipeline:
    """Fast hand tracking with ONNX Runtime."""

    def __init__(
        self,
        use_quantized=False,
        prune_ratio=0.0,
        precision='fp32',
        providers=None,
        detector_weights=None,
        landmark_weights=None,
        model_tag=None,
        build_only=False,
    ):
        self.base_path = str(MPT_DIR)
        self.model_dir = MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

        # Backwards compatibility: old --quantize flag maps to int8
        precision = precision.lower()
        if use_quantized and precision == 'fp32':
            precision = 'int8'
        if precision not in ('fp32', 'fp16', 'int8'):
            raise ValueError("precision must be one of: fp32, fp16, int8")

        self.precision = precision
        self.prune_ratio = prune_ratio
        self.providers = providers or _select_providers()
        self.model_tag = model_tag
        self.detector_weights = detector_weights or str(Path(self.base_path) / 'blazepalm.pth')
        self.landmark_weights = landmark_weights or str(Path(self.base_path) / 'blazehand_landmark.pth')

        # Prepare models
        self._prepare_models()

        det_path = str(self.model_dir / self._get_detector_name())
        lm_path = str(self.model_dir / self._get_landmark_name())
        anchors_path = str(Path(self.base_path) / 'anchors_palm.npy')

        if build_only:
            self.detector = None
            self.landmark = None
        else:
            self.detector = ONNXDetector(det_path, anchors_path, providers=self.providers)
            self.landmark = ONNXLandmark(lm_path, providers=self.providers)

        # Stats
        self.det_size = os.path.getsize(det_path) / (1024 * 1024)
        self.lm_size = os.path.getsize(lm_path) / (1024 * 1024)

    def _get_detector_name(self):
        name = 'palm_detector'
        if self.prune_ratio > 0:
            name += f'_pruned{int(self.prune_ratio*100)}'
        if self.precision == 'int8':
            name += '_int8'
        elif self.precision == 'fp16':
            name += '_fp16'
        if self.model_tag:
            name += f'_{self.model_tag}'
        return name + '.onnx'

    def _get_landmark_name(self):
        name = 'hand_landmark'
        if self.prune_ratio > 0:
            name += f'_pruned{int(self.prune_ratio*100)}'
        if self.precision == 'int8':
            name += '_int8'
        elif self.precision == 'fp16':
            name += '_fp16'
        if self.model_tag:
            name += f'_{self.model_tag}'
        return name + '.onnx'

    def _prepare_models(self):
        """Export/convert models if needed."""
        det_path = str(self.model_dir / self._get_detector_name())
        lm_path = str(self.model_dir / self._get_landmark_name())

        if os.path.exists(det_path) and os.path.exists(lm_path):
            print(f"Using cached models from {self.model_dir}")
            return

        print("Preparing ONNX models...")

        # Load PyTorch models
        detector = BlazePalm()
        detector.load_weights(self.detector_weights)
        detector.eval()

        landmark = BlazeHandLandmark()
        landmark.load_weights(self.landmark_weights)
        landmark.eval()

        # Apply pruning if requested
        if self.prune_ratio > 0:
            print(f"Applying {self.prune_ratio*100:.0f}% weight pruning...")
            self._apply_weight_pruning(detector, self.prune_ratio)
            self._apply_weight_pruning(landmark, self.prune_ratio)

        # Export to ONNX
        base_det = det_path.replace('_int8', '').replace('_fp16', '') if self.precision != 'fp32' else det_path
        base_lm = lm_path.replace('_int8', '').replace('_fp16', '') if self.precision != 'fp32' else lm_path

        print("Exporting detector to ONNX...")
        self._export_detector(detector, base_det)

        print("Exporting landmark to ONNX...")
        self._export_landmark(landmark, base_lm)

        # Quantize / convert precision if requested
        if self.precision == 'int8':
            print("Quantizing to INT8 (dynamic, CPU-friendly)...")
            quantize_dynamic(base_det, det_path, weight_type=QuantType.QUInt8)
            quantize_dynamic(base_lm, lm_path, weight_type=QuantType.QUInt8)
        elif self.precision == 'fp16':
            print("Converting weights to FP16 (Jetson-friendly)...")
            self._convert_to_fp16(base_det, det_path)
            self._convert_to_fp16(base_lm, lm_path)

        # Clean up fp32 exports if not the final artifact
        if self.precision != 'fp32':
            if base_det != det_path and os.path.exists(base_det):
                os.remove(base_det)
            if base_lm != lm_path and os.path.exists(base_lm):
                os.remove(base_lm)

        print("Models ready!")

    def _apply_weight_pruning(self, model, ratio):
        """Zero out smallest weights per layer."""
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                w = module.weight.data
                threshold = torch.quantile(w.abs().flatten(), ratio)
                mask = w.abs() >= threshold
                module.weight.data = w * mask.float()

    def _convert_to_fp16(self, src_path, dst_path):
        """Convert an ONNX model to FP16 while keeping IO types in FP32."""
        try:
            from onnxconverter_common import float16
        except ImportError as e:
            raise RuntimeError("onnxconverter_common is required for FP16 conversion. Install via pip.") from e

        model = onnx.load(src_path)
        model = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model, dst_path)

    def _export_detector(self, model, path):
        """Export detector to ONNX."""
        model.eval()
        dummy = torch.randn(1, 3, 256, 256)
        torch.onnx.export(
            model, dummy, path,
            input_names=['input'],
            output_names=['boxes', 'scores'],
            opset_version=13,
            do_constant_folding=True
        )

    def _export_landmark(self, model, path):
        """Export landmark model to ONNX with dynamic batch size."""
        model.eval()
        dummy = torch.randn(1, 3, 256, 256)
        torch.onnx.export(
            model, dummy, path,
            input_names=['input'],
            output_names=['flag', 'handedness', 'landmarks'],
            opset_version=13,
            do_constant_folding=True,
            dynamic_axes={'input': {0: 'batch'}, 'flag': {0: 'batch'},
                         'handedness': {0: 'batch'}, 'landmarks': {0: 'batch'}}
        )

    def process_frame(self, frame):
        """Process a single frame."""
        h, w = frame.shape[:2]
        frame_rgb = frame[:, :, ::-1].copy()

        # Resize for detector
        img256, _, scale, pad = resize_pad(frame_rgb)

        # Detect palms
        detections = self.detector.predict(img256)

        if len(detections) == 0:
            return [], [], scale, pad

        # Denormalize detections to image coordinates
        det_img = detections.copy()
        det_img[:, 0] = detections[:, 0] * scale * 256 - pad[0]
        det_img[:, 1] = detections[:, 1] * scale * 256 - pad[1]
        det_img[:, 2] = detections[:, 2] * scale * 256 - pad[0]
        det_img[:, 3] = detections[:, 3] * scale * 256 - pad[1]
        det_img[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]  # x coords
        det_img[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]  # y coords

        # Extract ROIs
        rois, affines = extract_roi(frame_rgb, det_img)

        if len(rois) == 0:
            return [], detections, scale, pad

        # Stack ROIs for batch inference
        roi_batch = np.stack(rois, axis=0)

        # Detect landmarks
        flags, _, landmarks = self.landmark.predict(roi_batch)

        # Denormalize landmarks
        landmarks_img = denormalize_landmarks(landmarks, affines)

        # Filter by confidence
        results = []
        for i, (flag, lm) in enumerate(zip(flags, landmarks_img)):
            if flag > 0.5:
                results.append(lm)

        return results, detections, scale, pad

    def run_webcam(self):
        """Run on webcam."""
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

            # Draw results
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
    # Generate dummy frames
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        pipeline.process_frame(dummy_frame)

    # Benchmark
    times = []
    for _ in range(num_frames):
        start = time.time()
        pipeline.process_frame(dummy_frame)
        times.append(time.time() - start)

    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'fps': 1000 / (np.mean(times) * 1000)
    }


def run_benchmarks():
    """Run all benchmark configurations."""
    print("\n" + "="*70)
    print("Benchmarking Hand Tracking Pipeline")
    print("="*70)

    configs = [
        {"name": "Original FP32", "precision": "fp32", "prune": 0.0},
        {"name": "FP16 (Jetson)", "precision": "fp16", "prune": 0.0},
        {"name": "INT8 (CPU)", "precision": "int8", "prune": 0.0},
        {"name": "Pruned 30% FP32", "precision": "fp32", "prune": 0.3},
        {"name": "Pruned 30% FP16", "precision": "fp16", "prune": 0.3},
        {"name": "Pruned 30% INT8", "precision": "int8", "prune": 0.3},
    ]

    results = []

    for cfg in configs:
        print(f"\nTesting: {cfg['name']}...")

        try:
            pipeline = HandTrackingPipeline(
                prune_ratio=cfg['prune'],
                precision=cfg['precision']
            )

            stats = benchmark(pipeline, num_frames=50)

            results.append({
                'name': cfg['name'],
                'det_size': pipeline.det_size,
                'lm_size': pipeline.lm_size,
                'total_size': pipeline.det_size + pipeline.lm_size,
                'mean_ms': stats['mean_ms'],
                'fps': stats['fps']
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Print results
    print("\n" + "="*85)
    print(f"{'Config':<25} {'Det Size':>10} {'LM Size':>10} {'Total':>10} {'Time':>12} {'FPS':>8}")
    print("="*85)

    baseline = results[0]['mean_ms'] if results else 1

    for r in results:
        speedup = baseline / r['mean_ms']
        print(f"{r['name']:<25} {r['det_size']:>8.2f}MB {r['lm_size']:>8.2f}MB "
              f"{r['total_size']:>8.2f}MB {r['mean_ms']:>10.2f}ms {r['fps']:>6.1f} ({speedup:.2f}x)")

    print("="*85)


# ============================================================
# Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--quantize', action='store_true',
                        help='Use INT8 quantization (legacy, same as --precision int8)')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp32',
                        help='Model precision. Use fp16 for Jetson/TensorRT, int8 for CPU-only.')
    parser.add_argument('--prune', type=float, default=0.0, help='Pruning ratio (0-1)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')

    args = parser.parse_args()

    precision = args.precision
    if args.quantize:
        precision = 'int8'

    if args.benchmark:
        run_benchmarks()
    else:
        pipeline = HandTrackingPipeline(
            use_quantized=args.quantize,
            prune_ratio=args.prune,
            precision=precision
        )
        pipeline.print_stats()
        pipeline.run_webcam()


if __name__ == "__main__":
    main()
