"""
Quick model comparison script.
Compares FP32, FP16, INT8, and pruned models on synthetic test data.
"""

import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import sys

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "assets/models"


def benchmark_model(model_path, num_frames=100):
    """
    Quick benchmark of a single model.

    Returns:
        dict with fps, latency, model_size, or None if model fails
    """
    if not model_path.exists():
        return None

    try:
        print(f"  Loading {model_path.name}...", end=" ", flush=True)
        model = YOLO(str(model_path))
        print("OK")

        # Generate test frames
        test_frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(10)
        ]

        # Warm-up
        for _ in range(5):
            _ = model(test_frames[0], verbose=False)

        # Benchmark
        times = []
        for i in range(num_frames):
            frame = test_frames[i % len(test_frames)]
            start = time.time()
            _ = model(frame, verbose=False)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        fps = 1.0 / avg_time

        return {
            'name': model_path.name,
            'fps': fps,
            'latency_ms': avg_time * 1000,
            'size_mb': model_path.stat().st_size / (1024 * 1024)
        }

    except Exception as e:
        print(f"FAILED ({e})")
        return None


def main():
    print("=" * 80)
    print("YOLO11n Hand Pose - Quick Model Comparison")
    print("=" * 80)

    # Define models to test
    model_configs = [
        ("FP32 (Baseline)", MODELS_DIR / "yolo11n_hand_pose.pt"),
        ("FP16 TensorRT", MODELS_DIR / "yolo11n_hand_pose_fp16.engine"),
        ("INT8 TensorRT", MODELS_DIR / "yolo11n_hand_pose_int8.engine"),
        ("ONNX", MODELS_DIR / "yolo11n_hand_pose.onnx"),
        ("Pruned 10%", MODELS_DIR / "yolo11n_hand_pose_pruned_10.pt"),
        ("Pruned 30%", MODELS_DIR / "yolo11n_hand_pose_pruned_30.pt"),
        ("Pruned 50%", MODELS_DIR / "yolo11n_hand_pose_pruned_50.pt"),
    ]

    results = []

    print("\nBenchmarking models (100 frames each)...\n")

    for name, path in model_configs:
        print(f"[{name}]")
        result = benchmark_model(path, num_frames=100)
        if result:
            results.append(result)
        print()

    if not results:
        print("No models found to benchmark!")
        print(f"\nMake sure you have at least the base model:")
        print(f"  {MODELS_DIR / 'yolo11n_hand_pose.pt'}")
        print(f"\nRun export_optimized_models.py to create optimized models.")
        return 1

    # Print results table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    header = f"{'Model':<25} {'FPS':<12} {'Latency (ms)':<15} {'Size (MB)':<12}"
    print(header)
    print("-" * 80)

    # Find baseline for speedup calculation
    baseline_fps = None
    for r in results:
        if 'yolo11n_hand_pose.pt' in r['name']:
            baseline_fps = r['fps']
            break

    for r in results:
        speedup = ""
        if baseline_fps and r['fps'] != baseline_fps:
            speedup = f"({r['fps']/baseline_fps:.2f}x)"

        row = f"{r['name']:<25} {r['fps']:<12.1f} {r['latency_ms']:<15.1f} {r['size_mb']:<12.2f}"
        if speedup:
            row += f"  {speedup}"
        print(row)

    print("=" * 80)

    # Recommendations
    print("\nRECOMMENDATIONS")
    print("-" * 80)

    if any('fp16' in r['name'].lower() for r in results):
        print("✓ FP16 TensorRT available - recommended for Jetson Nano")
    else:
        print("○ FP16 TensorRT not found - run export_optimized_models.py on Jetson Nano")

    if any('int8' in r['name'].lower() for r in results):
        print("✓ INT8 TensorRT available - best for maximum speed")
    else:
        print("○ INT8 TensorRT not found - run export_optimized_models.py on Jetson Nano")

    print("\nTo create missing models:")
    print("  python3 export_optimized_models.py")
    print("\nTo run the app with optimized model:")
    print("  python3 src/emoji_reactor/app.py --precision fp16")
    print("  python3 src/emoji_reactor/app.py --precision int8")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
