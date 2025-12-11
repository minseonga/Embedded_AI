
import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import sys

# Add src to path if needed, though usually automatic if running from root
sys.path.append(str(Path(__file__).parent))
try:
    from src.optimization import get_model_info
except ImportError:
    # Fallback if src is not found (e.g. running from wrong dir)
    print("Warning: could not import src.optimization. FLOPs calculation might fail.")
    def get_model_info(model, input_size=(1,3,640,640)):
        return 0, 0

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets/models/yolo11n_hand_pose.pt"
OUTPUT_DIR = ROOT / "assets/models"

class BenchmarkResults:
    def __init__(self):
        self.results = []

    def add(self, name, fps, macs_g, params_m, model_size_mb):
        self.results.append({
            'name': name,
            'fps': fps,
            'macs_g': macs_g,
            'params_m': params_m,
            'model_size_mb': model_size_mb
        })

    def print_table(self):
        print("\n" + "=" * 100)
        print(f"{'Model':<30} {'FPS':<10} {'GFLOPs':<10} {'Params(M)':<10} {'Size(MB)':<10}")
        print("=" * 100)
        for r in self.results:
            print(f"{r['name']:<30} {r['fps']:<10.1f} {r['macs_g']:<10.2f} {r['params_m']:<10.2f} {r['model_size_mb']:<10.2f}")
        print("=" * 100)


def get_model_size_mb(model_path):
    """Get model file size in MB."""
    if not Path(model_path).exists():
        return 0.0
    return Path(model_path).stat().st_size / (1024 * 1024)


def benchmark_inference(model_wrapper, test_frames, num_runs=50):
    """
    Benchmark inference speed.
    Args:
        model_wrapper: YOLO or equivalent callable
        test_frames: list of numpy images
    """
    times = []
    
    # Warmup
    print("  Warmup...", end="\r")
    for _ in range(10):
        _ = model_wrapper(test_frames[0], verbose=False)

    print("  Benchmarking...", end="\r")
    for i in range(num_runs):
        frame = test_frames[np.random.randint(len(test_frames))]
        start = time.time()
        _ = model_wrapper(frame, verbose=False)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    print(f"  Done. Avg latency: {avg_time*1000:.1f}ms")
    return fps


def prepare_test_frames(num_frames=20):
    """Generate test frames."""
    frames = []
    for _ in range(num_frames):
        # Random 640x480 RGB frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames.append(frame)
    return frames


def load_custom_pruned_model(path):
    """Load the custom pruned model checkpoint."""
    try:
        if not Path(path).exists():
            return None
        
        # Load checkpoint
        # TRUSTED LOCAL FILE: We created this file, so we disable the security check
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        
        # If it's a dict with 'model', extract it
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model = ckpt['model']
        else:
            model = ckpt # Assuming it's the model itself
            
        # Wrap in a simple callable to mimic YOLO class behavior for inference
        class PrunedWrapper:
            def __init__(self, model):
                self.model = model
                self.model.eval()
                
            def __call__(self, source, verbose=False):
                # source: numpy image HWC
                # Preprocess
                if isinstance(source, np.ndarray):
                     # Simple resize and normalize for benchmarking speed
                     # This is NOT correct preprocessing for accuracy, but OK for speed benchmark
                     img = cv2.resize(source, (640, 640))
                     img = img.transpose(2, 0, 1)
                     img = torch.from_numpy(img).float() / 255.0
                     img = img.unsqueeze(0)
                else:
                    img = source

                with torch.no_grad():
                    return self.model(img)
                    
        return PrunedWrapper(model), model
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None, None


def benchmark_all():
    print("=" * 80)
    print("YOLO11n Hand Pose - Real Optimization Benchmark")
    print("=" * 80)

    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    test_frames = prepare_test_frames()
    results = BenchmarkResults()

    # 1. Baseline
    print("\n[1] Benchmarking Baseline (FP32)...")
    model_baseline = YOLO(str(MODEL_PATH))
    fps_base = benchmark_inference(model_baseline, test_frames)
    
    # Calculate FLOPs/Params
    macs_base, params_base = get_model_info(model_baseline.model)
    size_base = get_model_size_mb(MODEL_PATH)
    
    results.add("Baseline (FP32)", fps_base, macs_base/1e9, params_base/1e6, size_base)

    # 2. Pruned Models
    print("\n[2] Benchmarking Pruned Models...")
    pruning_rates = [10, 30, 50, 70]
    
    for rate in pruning_rates:
        pruned_path = OUTPUT_DIR / f"yolo11n_hand_pose_pruned_{rate}.pt"
        if pruned_path.exists():
            print(f"  Testing {pruned_path.name}...")
            wrapper, model = load_custom_pruned_model(pruned_path)
            
            if wrapper:
                fps = benchmark_inference(wrapper, test_frames)
                macs, params = get_model_info(model)
                size = get_model_size_mb(pruned_path)
                
                results.add(f"Pruned {rate}%", fps, macs/1e9, params/1e6, size)
            else:
                print(f"  Could not load {pruned_path.name}")
        else:
            print(f"  {pruned_path.name} not found. Run export_optimized_models.py first.")

    # 3. Quantized (INT8 via ONNX)
    print("\n[3] Benchmarking INT8 (ONNX)...")
    int8_path = OUTPUT_DIR / "yolo11n_hand_pose_int8.onnx"
    
    if int8_path.exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(int8_path))
            
            # Simple ONNX benchmarking wrapper
            class ONNXWrapper:
                def __init__(self, sess):
                    self.sess = sess
                    self.input_name = sess.get_inputs()[0].name
                def __call__(self, source, verbose=False):
                    img = cv2.resize(source, (640, 640))
                    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                    img = np.expand_dims(img, 0)
                    self.sess.run(None, {self.input_name: img})

            wrapper = ONNXWrapper(session)
            fps_int8 = benchmark_inference(wrapper, test_frames)
            size_int8 = get_model_size_mb(int8_path)
            
            # FLOPs for ONNX is hard to count dynamically, use baseline or N/A
            # Quantization doesn't reduce FLOPs in theory (just precision), but speed increases.
            results.add("INT8 (ONNX)", fps_int8, macs_base/1e9, params_base/1e6, size_int8)
            
        except ImportError:
            print("onnxruntime not installed.")
    else:
        print(f"  {int8_path.name} not found.")

    results.print_table()


if __name__ == "__main__":
    benchmark_all()

