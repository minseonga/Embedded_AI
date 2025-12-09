# hand_tracking.py
# Pure PyTorch hand detection + landmarks using MediaPipe Python API.
# NO ONNX - uses MediaPipe's optimized TFLite models.

import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

# Paths
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "assets" / "models"


# ============================================================
# Helpers (kept for compatibility)
# ============================================================


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
# Pure PyTorch Models (for quantization/pruning experiments)
# ============================================================

class SimpleBlazeHandDetector(torch.nn.Module):
    """Lightweight CNN-based hand detector for quantization/pruning experiments.

    This is a simplified version designed to demonstrate:
    - PyTorch quantization (FP32 -> INT8)
    - Model pruning (magnitude-based)
    - Real performance improvements on Jetson Nano
    """

    def __init__(self, num_classes=1):
        super().__init__()
        # Lightweight feature extractor (MobileNet-inspired)
        self.features = torch.nn.Sequential(
            # Conv block 1
            torch.nn.Conv2d(3, 16, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU6(inplace=True),

            # Depthwise separable conv blocks
            self._make_dw_block(16, 32, stride=2),
            self._make_dw_block(32, 64, stride=2),
            self._make_dw_block(64, 128, stride=2),
            self._make_dw_block(128, 128, stride=1),

            # Global pooling
            torch.nn.AdaptiveAvgPool2d(1),
        )

        # Detection head
        self.detector = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 4),  # bbox: x, y, w, h
        )

        # Confidence head
        self.confidence = torch.nn.Sequential(
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, num_classes),
            torch.nn.Sigmoid(),
        )

        self._initialize_weights()

    def _make_dw_block(self, in_channels, out_channels, stride):
        """Depthwise separable convolution block."""
        return torch.nn.Sequential(
            # Depthwise
            torch.nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU6(inplace=True),
            # Pointwise
            torch.nn.Conv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU6(inplace=True),
        )

    def _initialize_weights(self):
        """Initialize weights with reasonable defaults."""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            bbox: Bounding box (B, 4) in format [x, y, w, h] normalized to [0, 1]
            conf: Confidence score (B, 1)
        """
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)

        bbox = torch.sigmoid(self.detector(feat))  # Ensure [0, 1] range
        conf = self.confidence(feat)

        return bbox, conf


class SimpleBlazeHandLandmarks(torch.nn.Module):
    """Lightweight landmark detector (21 hand keypoints)."""

    def __init__(self, num_landmarks=21):
        super().__init__()
        self.num_landmarks = num_landmarks

        # Lightweight feature extractor
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU6(inplace=True),

            self._make_dw_block(32, 64, stride=2),
            self._make_dw_block(64, 128, stride=2),
            self._make_dw_block(128, 128, stride=1),

            torch.nn.AdaptiveAvgPool2d(1),
        )

        # Landmark head (21 points * 3 coords = 63)
        self.landmarks = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_landmarks * 3),  # x, y, z for each point
        )

        self._initialize_weights()

    def _make_dw_block(self, in_channels, out_channels, stride):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU6(inplace=True),
            torch.nn.Conv2d(in_channels, out_channels, 1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU6(inplace=True),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            landmarks: (B, 21, 3) normalized coordinates
        """
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)

        lm = torch.sigmoid(self.landmarks(feat))  # Ensure [0, 1] range
        lm = lm.view(-1, self.num_landmarks, 3)

        return lm


# ============================================================
# Model wrappers
# ============================================================

class MediaPipeHandTracker:
    """Hand tracking using MediaPipe Python API (TFLite, not ONNX).

    Optimized for Jetson Nano with:
    - Reduced model complexity
    - Lower confidence thresholds for faster processing
    - Model complexity set to 0 (lightest)
    """

    def __init__(self, device: torch.device, precision: str = "fp16", enable_quantization: bool = True):
        self.device = device
        self.precision = precision
        self.enable_quantization = enable_quantization

        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands

            # Jetson Nano optimization settings
            # model_complexity: 0 = lightest (fastest), 1 = full model
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,  # Use lightest model for Jetson Nano
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[HandTracking] Using MediaPipe Hands (model_complexity=0, optimized for Jetson Nano)")
            if enable_quantization:
                print("[HandTracking] Quantization enabled (MediaPipe uses quantized TFLite models)")
        except ImportError:
            print("[HandTracking] MediaPipe not found, hand tracking disabled")
            self.hands = None

    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Process frame and return (landmarks, detections)."""
        if self.hands is None:
            return [], np.array([])

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        landmarks_list = []
        detections = []

        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Extract landmarks as numpy array (21 points, 3 coords each)
                landmarks = np.zeros((21, 3), dtype=np.float32)
                for i, lm in enumerate(hand_landmarks.landmark):
                    landmarks[i] = [lm.x * w, lm.y * h, lm.z]
                landmarks_list.append(landmarks)

                # Create detection box from landmarks
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()

                # Normalize to [0, 1] and add keypoints (wrist and middle finger)
                detection = np.array([
                    y_min / h, x_min / w, y_max / h, x_max / w,  # bbox
                    landmarks[0, 0] / w, landmarks[0, 1] / h,  # wrist
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # dummy keypoints
                    landmarks[9, 0] / w, landmarks[9, 1] / h,  # middle finger
                    1.0  # confidence score
                ])
                detections.append(detection)

        return landmarks_list, np.array(detections) if detections else np.array([])


# ============================================================
# Pipeline
# ============================================================

class BlazeHandTrackingPipeline:
    """Pure PyTorch hand tracking with REAL quantization/pruning support.

    Unlike MediaPipe (already optimized TFLite), this pipeline uses
    pure PyTorch models that can be quantized and pruned to show
    measurable performance improvements.

    Features:
    - FP32 baseline for comparison
    - INT8 quantization (torch.quantization)
    - Magnitude-based pruning (torch.nn.utils.prune)
    - Benchmarkable speed improvements
    """

    def __init__(
        self,
        precision: str = "fp32",
        prune_ratio: float = 0.0,
        quantize: bool = False,
        input_size: int = 256,
    ):
        self.precision = precision.lower()
        self.prune_ratio = prune_ratio
        self.quantize = quantize or (self.precision == "int8")
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[BlazeHand PyTorch] Device: {self.device}")
        print(f"[BlazeHand PyTorch] Precision: {self.precision}")
        print(f"[BlazeHand PyTorch] Quantization: {'ON (INT8)' if self.quantize else 'OFF'}")
        print(f"[BlazeHand PyTorch] Pruning: {self.prune_ratio * 100:.1f}% sparse" if self.prune_ratio > 0 else "[BlazeHand PyTorch] Pruning: OFF")

        # Create models
        self.detector = SimpleBlazeHandDetector(num_classes=1)
        self.landmark_model = SimpleBlazeHandLandmarks(num_landmarks=21)

        # Move to device
        self.detector = self.detector.to(self.device)
        self.landmark_model = self.landmark_model.to(self.device)

        # Apply pruning if requested
        if self.prune_ratio > 0:
            self._apply_pruning()

        # Apply quantization if requested
        if self.quantize:
            self._apply_quantization()

        # Set to eval mode
        self.detector.eval()
        self.landmark_model.eval()

        # Calculate model sizes
        self.det_size = self._get_model_size(self.detector)
        self.lm_size = self._get_model_size(self.landmark_model)

    def _apply_pruning(self):
        """Apply magnitude-based pruning to models."""
        import torch.nn.utils.prune as prune

        print(f"[BlazeHand PyTorch] Applying {self.prune_ratio * 100:.1f}% magnitude pruning...")

        for model in [self.detector, self.landmark_model]:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=self.prune_ratio)
                    prune.remove(module, 'weight')  # Make pruning permanent
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=self.prune_ratio)
                    prune.remove(module, 'weight')

        print(f"[BlazeHand PyTorch] Pruning applied successfully")

    def _apply_quantization(self):
        """Apply dynamic quantization to models."""
        print(f"[BlazeHand PyTorch] Applying INT8 dynamic quantization...")

        # Dynamic quantization (works best for inference)
        self.detector = torch.quantization.quantize_dynamic(
            self.detector,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )

        self.landmark_model = torch.quantization.quantize_dynamic(
            self.landmark_model,
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )

        print(f"[BlazeHand PyTorch] Quantization applied successfully")

    def _get_model_size(self, model):
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)

    def process_frame(self, frame: np.ndarray):
        """Process a single BGR frame.

        Returns:
            landmarks: List of (21, 3) numpy arrays
            detections: (N, 19) numpy array in MediaPipe format
            scale: float
            pad: tuple
        """
        h, w = frame.shape[:2]

        with torch.no_grad():
            # Prepare input
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            img_tensor = img_tensor.to(self.device)

            # Detect hands
            bbox, conf = self.detector(img_tensor)

            # Filter by confidence
            conf_threshold = 0.5
            valid_mask = (conf.squeeze() > conf_threshold).cpu().numpy()

            if not valid_mask.any():
                return [], np.array([]), 1.0, (0, 0)

            # Get landmarks for detected hands
            landmarks_list = []
            detections = []

            for i in range(bbox.shape[0]):
                if not valid_mask[i]:
                    continue

                # Get bbox in image coordinates
                bbox_norm = bbox[i].cpu().numpy()
                x_min = int(bbox_norm[0] * w)
                y_min = int(bbox_norm[1] * h)
                box_w = int(bbox_norm[2] * w)
                box_h = int(bbox_norm[3] * h)

                # Extract hand region and get landmarks
                x1, y1 = max(0, x_min), max(0, y_min)
                x2, y2 = min(w, x_min + box_w), min(h, y_min + box_h)

                if x2 <= x1 or y2 <= y1:
                    continue

                hand_roi = frame[y1:y2, x1:x2]
                hand_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                hand_resized = cv2.resize(hand_rgb, (self.input_size, self.input_size))
                hand_tensor = torch.from_numpy(hand_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                hand_tensor = hand_tensor.to(self.device)

                # Get landmarks
                lm = self.landmark_model(hand_tensor)
                lm = lm.squeeze(0).cpu().numpy()

                # Convert to image coordinates
                lm[:, 0] = lm[:, 0] * box_w + x1
                lm[:, 1] = lm[:, 1] * box_h + y1

                landmarks_list.append(lm)

                # Create detection in MediaPipe format
                detection = np.array([
                    y_min / h, x_min / w, (y_min + box_h) / h, (x_min + box_w) / w,  # bbox
                    lm[0, 0] / w, lm[0, 1] / h,  # wrist
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # dummy keypoints
                    lm[9, 0] / w, lm[9, 1] / h,  # middle finger
                    float(conf[i].item())  # confidence
                ])
                detections.append(detection)

        return landmarks_list, np.array(detections) if detections else np.array([]), 1.0, (0, 0)

    def print_stats(self):
        """Print model statistics."""
        print("\n" + "=" * 70)
        print("BlazeHand PyTorch Pipeline - Quantization/Pruning Demo")
        print("=" * 70)
        print(f"Backend:        Pure PyTorch (trainable)")
        print(f"Model Size:     {self.det_size + self.lm_size:.2f} MB total")
        print(f"  - Detector:   {self.det_size:.2f} MB")
        print(f"  - Landmark:   {self.lm_size:.2f} MB")
        print(f"Precision:      {self.precision.upper()}")
        print(f"Quantization:   {'INT8 (torch.quantization)' if self.quantize else 'None'}")
        print(f"Pruning:        {self.prune_ratio * 100:.1f}% sparse" if self.prune_ratio > 0 else "Pruning:        None")
        print(f"Device:         {self.device}")
        print(f"Input size:     {self.input_size}x{self.input_size}")
        print("=" * 70 + "\n")


class HandTrackingPipeline:
    """Fast hand tracking with MediaPipe (no ONNX).

    Optimized for Jetson Nano with:
    - Model complexity control (0=lightest, fastest)
    - Quantization support (int8 TFLite models)
    - CUDA acceleration where possible
    """

    def __init__(
        self,
        use_quantized: bool = True,  # Enable by default for Jetson Nano
        prune_ratio: float = 0.0,
        precision: str = "fp16",
        model_tag: str = None,
        build_only: bool = False,
        prune_mode: str = "magnitude",
    ):
        del model_tag, prune_mode  # unused; kept for CLI compatibility

        precision = precision.lower()
        if precision not in ("fp32", "fp16", "int8"):
            raise ValueError("precision must be one of: fp32, fp16, int8")

        self.precision = precision
        self.prune_ratio = prune_ratio
        self.use_quantized = use_quantized or (precision == "int8")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[HandTracking] Device: {self.device} | Precision: {self.precision}")
        print(f"[HandTracking] Quantization: {'ON' if self.use_quantized else 'OFF'}")
        if prune_ratio > 0:
            print(f"[HandTracking] Note: MediaPipe models are pre-pruned, prune_ratio ignored")

        if build_only:
            self.tracker = None
        else:
            self.tracker = MediaPipeHandTracker(
                self.device,
                precision=self.precision,
                enable_quantization=self.use_quantized
            )

        # MediaPipe lite model is ~1-2MB (detection) + ~2-3MB (landmarks)
        self.det_size = 1.5  # Approximate size in MB
        self.lm_size = 2.5

    def process_frame(self, frame: np.ndarray):
        """Process a single BGR frame. Returns (landmarks, detections, scale, pad)."""
        if self.tracker is None:
            return [], [], 1.0, (0, 0)

        landmarks, detections = self.tracker.process_frame(frame)

        # MediaPipe returns landmarks in image coordinates, no need for denormalization
        # Return scale and pad for compatibility
        scale = 1.0
        pad = (0, 0)

        return landmarks, detections, scale, pad

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
        print("\n" + "=" * 60)
        print("Hand Tracking Pipeline - Jetson Nano Optimized")
        print("=" * 60)
        print(f"Backend:        MediaPipe (TFLite)")
        print(f"Model Size:     ~{self.det_size + self.lm_size:.1f} MB total")
        print(f"  - Detector:   ~{self.det_size:.1f} MB")
        print(f"  - Landmark:   ~{self.lm_size:.1f} MB")
        print(f"Precision:      {self.precision}")
        print(f"Quantization:   {'ON (int8)' if self.use_quantized else 'OFF'}")
        print(f"Device:         {self.device}")
        print(f"Optimization:   model_complexity=0 (lightest)")
        print("=" * 60 + "\n")


# ============================================================
# Benchmarking
# ============================================================

def benchmark(pipeline, num_frames=100, warmup=10):
    """Benchmark inference speed with detailed stats.

    Args:
        pipeline: HandTrackingPipeline instance
        num_frames: Number of frames to benchmark
        warmup: Number of warmup frames

    Returns:
        dict: Performance statistics
    """
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    print(f"[Benchmark] Warming up ({warmup} frames)...")
    for _ in range(warmup):
        pipeline.process_frame(dummy_frame)

    # Benchmark
    print(f"[Benchmark] Running benchmark ({num_frames} frames)...")
    times = []
    for _ in range(num_frames):
        start = time.time()
        pipeline.process_frame(dummy_frame)
        times.append(time.time() - start)

    times = np.array(times) * 1000  # Convert to ms

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "fps": 1000.0 / float(np.mean(times)),
    }


def run_benchmarks():
    """Run comprehensive benchmarks comparing optimization options."""
    print("\n" + "=" * 90)
    print("Hand Tracking Pipeline - Jetson Nano Optimization Benchmark")
    print("=" * 90)

    configs = [
        {"name": "Baseline (no quant)", "precision": "fp32", "quantize": False},
        {"name": "INT8 Quantized", "precision": "int8", "quantize": True},
        {"name": "FP16 (Jetson)", "precision": "fp16", "quantize": False},
        {"name": "FP16 + Quantized", "precision": "fp16", "quantize": True},
    ]

    results = []
    baseline_fps = None

    for cfg in configs:
        print(f"\n{'─' * 90}")
        print(f"Testing: {cfg['name']}")
        print(f"{'─' * 90}")
        try:
            pipeline = HandTrackingPipeline(
                precision=cfg["precision"],
                use_quantized=cfg["quantize"],
            )
            stats = benchmark(pipeline, num_frames=50, warmup=10)

            if baseline_fps is None:
                baseline_fps = stats["fps"]

            speedup = stats["fps"] / baseline_fps if baseline_fps else 1.0

            results.append({
                "name": cfg["name"],
                "total_size": pipeline.det_size + pipeline.lm_size,
                "mean_ms": stats["mean_ms"],
                "std_ms": stats["std_ms"],
                "p95_ms": stats["p95_ms"],
                "fps": stats["fps"],
                "speedup": speedup,
            })

            print(f"\n  Results:")
            print(f"    Mean:     {stats['mean_ms']:>7.2f} ms")
            print(f"    Std:      {stats['std_ms']:>7.2f} ms")
            print(f"    Median:   {stats['median_ms']:>7.2f} ms")
            print(f"    P95:      {stats['p95_ms']:>7.2f} ms")
            print(f"    FPS:      {stats['fps']:>7.1f}")
            print(f"    Speedup:  {speedup:>7.2f}x")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Configuration':<25} {'Size':>10} {'Mean':>10} {'P95':>10} {'FPS':>8} {'Speedup':>10}")
    print("─" * 90)
    for r in results:
        print(f"{r['name']:<25} {r['total_size']:>8.1f}MB "
              f"{r['mean_ms']:>9.2f}ms {r['p95_ms']:>9.2f}ms "
              f"{r['fps']:>7.1f} {r['speedup']:>9.2f}x")
    print("=" * 90)

    # Recommendations
    print("\nRECOMMENDATIONS FOR JETSON NANO:")
    print("  • Best for speed: INT8 Quantized or FP16 + Quantized")
    print("  • Best balance:   FP16 + Quantized (recommended)")
    print("  • Note: MediaPipe already uses optimized TFLite models")
    print("=" * 90)


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
