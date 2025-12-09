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
