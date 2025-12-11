# YOLO11n Hand Pose - Jetson Nano Optimization Guide

## Overview

This guide covers model optimization for Jetson Nano deployment:
- **TensorRT INT8**: ~2x size reduction, ~10-15% speed improvement, ~2% accuracy loss
- **TensorRT FP16**: ~50% size reduction, ~20-30% speed improvement, minimal accuracy loss
- **Structured Pruning**: Additional size/speed improvements with accuracy trade-offs

---

## Prerequisites

### On Jetson Nano:
```bash
# CUDA and TensorRT should already be installed with JetPack
python3 -m pip install ultralytics
python3 -m pip install opencv-python
python3 -m pip install numpy
```

### On Development Machine (optional):
```bash
pip3 install ultralytics onnx onnxruntime
```

---

## Step 1: Export Optimized Models

### On Jetson Nano (Recommended):

```bash
# Export all optimized models
python3 export_optimized_models.py
# Select option 5 (All of the above)
```

This will create:
- `yolo11n_hand_pose_int8.engine` - TensorRT INT8
- `yolo11n_hand_pose_fp16.engine` - TensorRT FP16
- `yolo11n_hand_pose.onnx` - ONNX (portable)
- `yolo11n_hand_pose_pruned_*.pt` - Pruned models (10%, 30%, 50%, 70%)

**Note**: TensorRT engines (.engine) are platform-specific. Export on the device where you'll run inference.

### On Development Machine:

You can export ONNX and pruned models, but TensorRT engines must be created on Jetson Nano.

```bash
python3 export_optimized_models.py
# Select option 3 (Pruned models) or 4 (ONNX)
```

---

## Step 2: Benchmark Performance

### Real-time Performance Test:

```bash
# Benchmark all available models on camera
python3 benchmark_realtime.py --source 0 --frames 300

# Benchmark specific models
python3 benchmark_realtime.py --models yolo11n_hand_pose.pt yolo11n_hand_pose_int8.engine --frames 300

# Benchmark on video file
python3 benchmark_realtime.py --source test_video.mp4 --frames 500
```

### Optimization Comparison:

```bash
# Compare quantization and pruning effects
python3 benchmark_optimization.py
```

---

## Step 3: Run Application with Optimized Models

### FP32 (Baseline):
```bash
python3 src/emoji_reactor/app.py --precision fp32
```

### FP16 (Recommended for Jetson Nano):
```bash
python3 src/emoji_reactor/app.py --precision fp16
```

### INT8 (Best Size/Speed):
```bash
python3 src/emoji_reactor/app.py --precision int8
```

---

## Performance Expectations

### Jetson Nano (4GB):

| Model | FPS (expected) | Size | Accuracy |
|-------|----------------|------|----------|
| FP32 (baseline) | 15-20 FPS | 6.0 MB | 100% |
| FP16 TensorRT | 20-25 FPS | 3.0 MB | 99.9% |
| INT8 TensorRT | 25-30 FPS | 1.5 MB | 98% |
| 30% Pruned + INT8 | 30-35 FPS | 1.2 MB | 85% |

**Target**: >20 FPS for smooth real-time operation

---

## Troubleshooting

### TensorRT Export Fails:
```bash
# Check CUDA is available
python3 -c "import torch; print(torch.cuda.is_available())"

# Check TensorRT installation
python3 -c "import tensorrt; print(tensorrt.__version__)"

# Install pycuda if missing
pip3 install pycuda
```

### Low Accuracy After INT8:
- Try FP16 instead if accuracy drop is unacceptable
- INT8 may have ~2% accuracy loss without calibration

### Low FPS on Jetson Nano:
- Ensure power mode is MAXN: `sudo nvpmodel -m 0`
- Increase clock speeds: `sudo jetson_clocks`
- Use INT8 or FP16 TensorRT engines
- Reduce camera resolution if needed

### Model Not Found:
```bash
# Check model files exist
ls -lh assets/models/

# Re-download base model if needed
wget https://github.com/chrismuntean/YOLO11n-pose-hands/raw/main/runs/pose/train/weights/best.pt -O assets/models/yolo11n_hand_pose.pt
```

---

## Recommendations

### For Jetson Nano:
1. **Start with FP16**: Best balance of speed, size, and accuracy
2. **Try INT8**: If you need maximum FPS and can tolerate ~2% accuracy loss
3. **Avoid heavy pruning**: >30% pruning significantly degrades hand detection quality
4. **Use GStreamer**: Better camera performance than OpenCV VideoCapture

### For Desktop/Laptop:
1. **Use FP32**: Your GPU has enough power, no need for optimization
2. **Or use ONNX**: Good cross-platform compatibility

### Production Deployment:
1. Test all precision levels with your specific use case
2. Benchmark with actual camera/lighting conditions
3. Validate accuracy on representative test set
4. Monitor temperature and power consumption

---

## Next Steps

1. Run `export_optimized_models.py` on Jetson Nano
2. Benchmark with `benchmark_realtime.py`
3. Choose best precision for your use case
4. Deploy with `src/emoji_reactor/app.py --precision [fp16|int8]`

For questions or issues, check the benchmark results and adjust accordingly.
