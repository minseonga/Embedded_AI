"""
Real-time performance benchmark
Measures actual FPS, latency, and accuracy on live camera feed or test video
"""

import time
import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def benchmark_model(model_path, test_source=0, num_frames=300):
    """
    Benchmark a YOLO model on real-time inference.

    Args:
        model_path: Path to model file
        test_source: Camera index or video file path
        num_frames: Number of frames to process

    Returns:
        dict with fps, latency, and other metrics
    """
    print(f"\nBenchmarking: {model_path.name}")
    print("-" * 60)

    # Load model
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    # Open video source
    cap = cv2.VideoCapture(test_source)
    if not cap.isOpened():
        print(f"Failed to open video source: {test_source}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm-up
    print("Warming up...")
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            _ = model(frame, verbose=False)

    # Benchmark
    print(f"Processing {num_frames} frames...")

    times = []
    num_detections = []
    keypoint_counts = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            # Loop video or generate random frame
            if isinstance(test_source, int):
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        start = time.time()
        results = model(frame, verbose=False)
        elapsed = time.time() - start

        times.append(elapsed)

        # Count detections
        if len(results) > 0:
            result = results[0]
            num_det = len(result.boxes) if result.boxes is not None else 0
            num_detections.append(num_det)

            # Count keypoints
            if result.keypoints is not None and len(result.keypoints) > 0:
                kpts = result.keypoints.data
                keypoint_counts.append(len(kpts))
            else:
                keypoint_counts.append(0)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{num_frames} frames processed...")

    cap.release()

    # Calculate metrics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    fps = 1.0 / avg_time
    fps_min = 1.0 / max_time
    fps_max = 1.0 / min_time

    results = {
        'model_name': model_path.name,
        'model_size_mb': model_path.stat().st_size / (1024 * 1024),
        'fps_avg': fps,
        'fps_min': fps_min,
        'fps_max': fps_max,
        'latency_avg_ms': avg_time * 1000,
        'latency_std_ms': std_time * 1000,
        'latency_min_ms': min_time * 1000,
        'latency_max_ms': max_time * 1000,
        'avg_detections': np.mean(num_detections) if num_detections else 0,
        'avg_keypoints': np.mean(keypoint_counts) if keypoint_counts else 0,
        'frames_processed': len(times)
    }

    return results


def print_results(results_list):
    """Print benchmark results in a table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    header = f"{'Model':<30} {'Size (MB)':<12} {'FPS (avg)':<12} {'FPS (min-max)':<18} {'Latency (ms)':<15}"
    print(header)
    print("-" * 100)

    for r in results_list:
        if r is None:
            continue

        fps_range = f"{r['fps_min']:.1f} - {r['fps_max']:.1f}"
        latency = f"{r['latency_avg_ms']:.1f} ± {r['latency_std_ms']:.1f}"

        row = f"{r['model_name']:<30} {r['model_size_mb']:<12.2f} {r['fps_avg']:<12.1f} {fps_range:<18} {latency:<15}"
        print(row)

    print("=" * 100)

    # Print detailed stats
    print("\nDetailed Statistics:")
    for r in results_list:
        if r is None:
            continue

        print(f"\n{r['model_name']}:")
        print(f"  Frames processed: {r['frames_processed']}")
        print(f"  Avg detections/frame: {r['avg_detections']:.2f}")
        print(f"  Avg keypoints/frame: {r['avg_keypoints']:.2f}")
        print(f"  Min latency: {r['latency_min_ms']:.2f} ms")
        print(f"  Max latency: {r['latency_max_ms']:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Real-time YOLO model benchmark")
    parser.add_argument('--source', type=str, default='0', help='Video source (camera index or file path)')
    parser.add_argument('--frames', type=int, default=300, help='Number of frames to process')
    parser.add_argument('--models', type=str, nargs='+', help='Specific model files to benchmark')
    args = parser.parse_args()

    # Parse video source
    try:
        test_source = int(args.source)
    except ValueError:
        test_source = args.source

    models_dir = ROOT / "assets/models"

    # Find models to benchmark
    if args.models:
        model_paths = [models_dir / m for m in args.models]
    else:
        # Auto-detect available models
        model_paths = []

        # Check for base model
        base_model = models_dir / "yolo11n_hand_pose.pt"
        if base_model.exists():
            model_paths.append(base_model)

        # Check for optimized models
        for suffix in ['_int8.engine', '_fp16.engine', '.onnx']:
            opt_model = models_dir / f"yolo11n_hand_pose{suffix}"
            if opt_model.exists():
                model_paths.append(opt_model)

        # Check for pruned models
        for rate in [10, 30, 50, 70]:
            pruned_model = models_dir / f"yolo11n_hand_pose_pruned_{rate}.pt"
            if pruned_model.exists():
                model_paths.append(pruned_model)

    if not model_paths:
        print("No models found to benchmark!")
        print(f"Looked in: {models_dir}")
        return

    print("=" * 100)
    print("YOLO11n Hand Pose - Real-time Performance Benchmark")
    print("=" * 100)
    print(f"\nVideo source: {test_source}")
    print(f"Frames to process: {args.frames}")
    print(f"Models to benchmark: {len(model_paths)}")

    for p in model_paths:
        print(f"  - {p.name}")

    print("\nStarting benchmark...")

    # Run benchmarks
    results_list = []
    for model_path in model_paths:
        result = benchmark_model(model_path, test_source, args.frames)
        results_list.append(result)

    # Print results
    print_results(results_list)

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS FOR JETSON NANO")
    print("=" * 100)

    if results_list:
        # Find best FPS
        valid_results = [r for r in results_list if r is not None]
        if valid_results:
            best_fps = max(valid_results, key=lambda x: x['fps_avg'])
            best_size = min(valid_results, key=lambda x: x['model_size_mb'])

            print(f"\n✓ Best FPS: {best_fps['model_name']} ({best_fps['fps_avg']:.1f} FPS)")
            print(f"✓ Smallest model: {best_size['model_name']} ({best_size['model_size_mb']:.2f} MB)")

            print("\nFor real-time performance on Jetson Nano:")
            print("  - Target: >20 FPS for smooth operation")
            print("  - TensorRT INT8: Best speed/size trade-off")
            print("  - TensorRT FP16: Good balance if INT8 accuracy drop is too high")
            print("  - Avoid heavy pruning (>50%) to maintain detection quality")

    print("=" * 100)


if __name__ == "__main__":
    main()
