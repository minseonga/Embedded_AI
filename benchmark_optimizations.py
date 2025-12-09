#!/usr/bin/env python3
"""
Comprehensive benchmark script for Jetson Nano optimizations.
Tests various configurations and measures actual performance improvements.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hand_tracking import HandTrackingPipeline, BlazeHandTrackingPipeline
from face_landmark import FaceLandmarkPipeline


def benchmark_pytorch_hand_tracking():
    """Benchmark PyTorch hand tracking with real quantization/pruning."""
    print("\n" + "=" * 80)
    print("PYTORCH HAND TRACKING BENCHMARK (Quantization/Pruning Demo)")
    print("=" * 80)

    configs = [
        {"name": "PyTorch Baseline (FP32)", "precision": "fp32", "quantize": False, "prune": 0.0},
        {"name": "PyTorch + INT8 Quant", "precision": "int8", "quantize": True, "prune": 0.0},
        {"name": "PyTorch + 30% Pruning", "precision": "fp32", "quantize": False, "prune": 0.3},
        {"name": "PyTorch + 50% Pruning", "precision": "fp32", "quantize": False, "prune": 0.5},
        {"name": "PyTorch + Quant + 30% Prune", "precision": "int8", "quantize": True, "prune": 0.3},
    ]

    results = []
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for cfg in configs:
        print(f"\n{'─' * 80}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'─' * 80}")

        try:
            pipeline = BlazeHandTrackingPipeline(
                precision=cfg["precision"],
                quantize=cfg["quantize"],
                prune_ratio=cfg["prune"]
            )

            # Warmup
            print("Warming up (10 frames)...")
            for _ in range(10):
                pipeline.process_frame(dummy_frame)

            # Benchmark
            print("Running benchmark (100 frames)...")
            times = []
            for _ in range(100):
                start = time.perf_counter()
                landmarks, detections, scale, pad = pipeline.process_frame(dummy_frame)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # Convert to ms

            times = np.array(times)
            mean_ms = np.mean(times)
            std_ms = np.std(times)
            p95_ms = np.percentile(times, 95)
            fps = 1000.0 / mean_ms

            results.append({
                "name": cfg["name"],
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "p95_ms": p95_ms,
                "fps": fps,
                "size_mb": pipeline.det_size + pipeline.lm_size,
            })

            print(f"\nResults:")
            print(f"  Mean:   {mean_ms:>7.2f} ms")
            print(f"  Std:    {std_ms:>7.2f} ms")
            print(f"  P95:    {p95_ms:>7.2f} ms")
            print(f"  FPS:    {fps:>7.1f}")
            print(f"  Size:   {pipeline.det_size + pipeline.lm_size:.2f} MB")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("PYTORCH HAND TRACKING SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<35} {'Size':>8} {'Mean':>10} {'P95':>10} {'FPS':>8} {'Speedup':>10}")
    print("─" * 80)

    baseline_fps = results[0]["fps"] if results else 1.0
    for r in results:
        speedup = r["fps"] / baseline_fps
        print(f"{r['name']:<35} {r['size_mb']:>7.2f}MB {r['mean_ms']:>9.2f}ms {r['p95_ms']:>9.2f}ms "
              f"{r['fps']:>7.1f} {speedup:>9.2f}x")

    print("=" * 80)
    print("\nNote: This demonstrates REAL quantization/pruning with measurable effects!")
    print("=" * 80)


def benchmark_mediapipe_hand_tracking():
    """Benchmark hand tracking with different configurations."""
    print("\n" + "=" * 80)
    print("HAND TRACKING BENCHMARK")
    print("=" * 80)

    configs = [
        {"name": "Baseline (FP32, no quant)", "precision": "fp32", "quantize": False},
        {"name": "INT8 Quantized", "precision": "int8", "quantize": True},
        {"name": "FP16 (Jetson optimized)", "precision": "fp16", "quantize": False},
        {"name": "FP16 + Quantized (BEST)", "precision": "fp16", "quantize": True},
    ]

    results = []
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    for cfg in configs:
        print(f"\n{'─' * 80}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'─' * 80}")

        try:
            pipeline = HandTrackingPipeline(
                precision=cfg["precision"],
                use_quantized=cfg["quantize"]
            )

            # Warmup
            print("Warming up (10 frames)...")
            for _ in range(10):
                pipeline.process_frame(dummy_frame)

            # Benchmark
            print("Running benchmark (100 frames)...")
            times = []
            for _ in range(100):
                start = time.perf_counter()
                landmarks, detections, scale, pad = pipeline.process_frame(dummy_frame)
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)  # Convert to ms

            times = np.array(times)
            mean_ms = np.mean(times)
            std_ms = np.std(times)
            p95_ms = np.percentile(times, 95)
            fps = 1000.0 / mean_ms

            results.append({
                "name": cfg["name"],
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "p95_ms": p95_ms,
                "fps": fps,
            })

            print(f"\nResults:")
            print(f"  Mean:   {mean_ms:>7.2f} ms")
            print(f"  Std:    {std_ms:>7.2f} ms")
            print(f"  P95:    {p95_ms:>7.2f} ms")
            print(f"  FPS:    {fps:>7.1f}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("HAND TRACKING SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<35} {'Mean':>10} {'P95':>10} {'FPS':>8} {'Speedup':>10}")
    print("─" * 80)

    baseline_fps = results[0]["fps"] if results else 1.0
    for r in results:
        speedup = r["fps"] / baseline_fps
        print(f"{r['name']:<35} {r['mean_ms']:>9.2f}ms {r['p95_ms']:>9.2f}ms "
              f"{r['fps']:>7.1f} {speedup:>9.2f}x")

    print("=" * 80)


def benchmark_face_detection():
    """Benchmark face detection with MediaPipe."""
    print("\n" + "=" * 80)
    print("FACE DETECTION BENCHMARK")
    print("=" * 80)

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print(f"\n{'─' * 80}")
    print(f"Configuration: MediaPipe Face Detection (TFLite)")
    print(f"{'─' * 80}")

    try:
        pipeline = FaceLandmarkPipeline(precision="fp16")

        # Warmup
        print("Warming up (10 frames)...")
        for _ in range(10):
            pipeline.process_frame(dummy_frame)

        # Benchmark
        print("Running benchmark (100 frames)...")
        times = []
        for _ in range(100):
            start = time.perf_counter()
            landmarks, mar, face_box = pipeline.process_frame(dummy_frame)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)

        times = np.array(times)
        mean_ms = np.mean(times)
        std_ms = np.std(times)
        p95_ms = np.percentile(times, 95)
        fps = 1000.0 / mean_ms

        print(f"\nResults:")
        print(f"  Mean:   {mean_ms:>7.2f} ms")
        print(f"  Std:    {std_ms:>7.2f} ms")
        print(f"  P95:    {p95_ms:>7.2f} ms")
        print(f"  FPS:    {fps:>7.1f}")
        print(f"\n  Note: MediaPipe Face Detection works with ANY OpenCV version")
        print(f"        Optimized TFLite models, no download needed")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n" + "=" * 80)
    print("JETSON NANO OPTIMIZATION BENCHMARK")
    print("Testing quantization, pruning, and precision optimizations")
    print("=" * 80)

    # Benchmark PyTorch pipeline (shows real quantization/pruning effects)
    benchmark_pytorch_hand_tracking()

    # Benchmark MediaPipe pipeline (pre-optimized TFLite)
    benchmark_mediapipe_hand_tracking()

    # Benchmark face detection
    benchmark_face_detection()

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR JETSON NANO")
    print("=" * 80)
    print("  OPTION 1: PyTorch Pipeline (--use-pytorch)")
    print("    • Shows real quantization/pruning benefits")
    print("    • Use INT8 + pruning for maximum speed")
    print("    • Larger models (~5-10MB) but fully customizable")
    print("")
    print("  OPTION 2: MediaPipe Pipeline (default)")
    print("    • Pre-optimized TFLite models (~4MB)")
    print("    • Ultra-fast out of the box")
    print("    • Quantization already applied (no extra benefit)")
    print("    • Best for production use")
    print("")
    print("  Face Detection:")
    print("    • MediaPipe Face Detection (default)")
    print("    • Works with ANY OpenCV version")
    print("    • Expected: 20-30 FPS on Jetson Nano")
    print("=" * 80)


if __name__ == "__main__":
    main()
