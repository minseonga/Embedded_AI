# Emoji Reactor - Hand & Face Tracking for Jetson Nano

Real-time emoji reaction app using hand pose and facial expression detection, optimized for NVIDIA Jetson Nano.

## Features

- **Hand Tracking**: YOLO11n-Pose (21 keypoints per hand)
- **Face Tracking**: MediaPipe Face Mesh (468 keypoints, focus on mouth)
- **3 Emoji States**:
  - 🙌 **HANDS_UP**: Hands raised above threshold
  - 😊 **SMILING**: Mouth aspect ratio > smile threshold
  - 😐 **STRAIGHT_FACE**: Default state
- **Audio**: Background music + sound effects on state change
- **Optimization**: TensorRT INT8/FP16, model pruning for Jetson Nano

## Quick Start

### Installation

```bash
# Clone the repository
cd /Users/medicalissue/Desktop/Embedded_AI

# Install dependencies
pip3 install ultralytics opencv-python mediapipe numpy

# Download model (if not present)
# Model is included in assets/models/yolo11n_hand_pose.pt
```

### Run the App

```bash
# On PC/Mac (no GStreamer)
python3 src/emoji_reactor/app.py --no-gstreamer --camera 0

# On Jetson Nano (with GStreamer)
python3 src/emoji_reactor/app.py

# With optimized model (FP16)
python3 src/emoji_reactor/app.py --precision fp16

# With INT8 quantization
python3 src/emoji_reactor/app.py --precision int8
```

### Command-Line Arguments

```
--precision {fp32,fp16,int8}  Model precision (default: fp32)
--camera INT                   Camera device ID (default: 0)
--raise-thresh FLOAT           Hand raise threshold (default: 0.25)
--smile-thresh FLOAT           Smile threshold (default: 0.35)
--no-mirror                    Disable horizontal flip
--no-gstreamer                 Disable GStreamer (use for PC/Mac)
```

## Project Structure

```
Embedded_AI/
├── src/
│   ├── hand_tracking/
│   │   ├── __init__.py
│   │   └── pipeline.py         # YOLO11n + MediaPipe pipeline
│   └── emoji_reactor/
│       └── app.py               # Main application
├── assets/
│   ├── models/
│   │   ├── yolo11n_hand_pose.pt              # Base model (6.0 MB)
│   │   ├── yolo11n_hand_pose_fp16.engine     # FP16 TensorRT
│   │   └── yolo11n_hand_pose_int8.engine     # INT8 TensorRT
│   ├── emojis/
│   │   ├── smile.jpg
│   │   ├── plain.png
│   │   └── air.jpg
│   └── audio/
│       ├── yessir.mp3           # Background music
│       ├── HANDS_UP.mp3         # State change sounds
│       ├── SMILING.mp3
│       └── STRAIGHT_FACE.mp3
├── export_optimized_models.py   # Create TensorRT/pruned models
├── benchmark_realtime.py        # Real-time performance test
├── benchmark_optimization.py    # Quantization/pruning comparison
├── compare_models.py            # Quick model comparison
├── OPTIMIZATION_GUIDE.md        # Detailed optimization guide
└── README.md                    # This file
```

## Model Optimization for Jetson Nano

### Performance Targets

| Model | FPS (Jetson Nano) | Size | Accuracy |
|-------|-------------------|------|----------|
| FP32 (baseline) | 15-20 FPS | 6.0 MB | 100% |
| FP16 TensorRT | 20-25 FPS | 3.0 MB | 99.9% |
| INT8 TensorRT | 25-30 FPS | 1.5 MB | 98% |

### Export Optimized Models

**On Jetson Nano** (recommended):

```bash
# Export all optimized models
python3 export_optimized_models.py
# Select option 5 (All of the above)
```

This creates:
- TensorRT INT8 engine (best speed)
- TensorRT FP16 engine (best balance)
- ONNX model (portable)
- Pruned models (10%, 30%, 50%, 70%)

**Note**: TensorRT engines are platform-specific. Export on the device where you'll run inference.

### Benchmark Performance

```bash
# Compare all available models
python3 compare_models.py

# Real-time benchmark on camera
python3 benchmark_realtime.py --source 0 --frames 300

# Quantization/pruning comparison
python3 benchmark_optimization.py
```

### Recommendations

- **Jetson Nano**: Use FP16 or INT8 TensorRT for best performance
- **Desktop/Laptop**: Use FP32 (no optimization needed)
- **INT8**: May have ~2% accuracy loss without calibration
- **Pruning**: Avoid >30% pruning to maintain hand detection quality

See [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) for detailed instructions.

## Technical Details

### Hand Detection: YOLO11n-Pose

- **Model**: chrismuntean/YOLO11n-pose-hands
- **Architecture**: YOLO11n with pose estimation head
- **Keypoints**: 21 per hand (MediaPipe format)
- **Training**: Hand Keypoint Dataset 26K
- **Input**: 640x480 RGB image
- **Output**: Bounding boxes + 21 (x, y, confidence) keypoints

### Face Detection: MediaPipe Face Mesh

- **Keypoints**: 468 facial landmarks
- **Focus**: Mouth region for smile detection
- **MAR Calculation**: Mouth height / mouth width
- **Indices Used**:
  - Upper lip: 13
  - Lower lip: 14
  - Left corner: 78
  - Right corner: 308

### State Detection Logic

```python
state = "STRAIGHT_FACE"  # Default

# Check if any hand wrist is above threshold
if any(hand.wrist.y / frame_height < raise_thresh):
    state = "HANDS_UP"

# Check if mouth aspect ratio indicates smile
elif mouth_aspect_ratio > smile_thresh:
    state = "SMILING"
```

## Development

### Adding New Emoji States

1. Add emoji image to `assets/emojis/`
2. Add sound effect to `assets/audio/`
3. Update `load_emojis()` in `app.py`
4. Add state detection logic in main loop

### Adjusting Thresholds

```bash
# Make hand detection more sensitive
python3 src/emoji_reactor/app.py --raise-thresh 0.35

# Make smile detection less sensitive
python3 src/emoji_reactor/app.py --smile-thresh 0.4
```

## Troubleshooting

### Camera Issues

```bash
# Check camera device
ls /dev/video*

# Try different camera ID
python3 src/emoji_reactor/app.py --camera 1

# Jetson Nano: Enable GStreamer
python3 src/emoji_reactor/app.py  # (no --no-gstreamer)
```

### Low FPS on Jetson Nano

```bash
# Set MAXN power mode
sudo nvpmodel -m 0

# Increase clock speeds
sudo jetson_clocks

# Use optimized model
python3 src/emoji_reactor/app.py --precision fp16
```

### TensorRT Export Fails

```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Install missing dependencies
pip3 install pycuda

# Export on Jetson Nano (platform-specific)
```

### Audio Not Playing

```bash
# Install ffplay (Linux)
sudo apt-get install ffmpeg

# macOS uses afplay (built-in)

# Check audio files exist
ls assets/audio/
```

## Credits

- **YOLO11n Hand Pose**: [chrismuntean/YOLO11n-pose-hands](https://github.com/chrismuntean/YOLO11n-pose-hands)
- **MediaPipe**: [Google MediaPipe](https://google.github.io/mediapipe/)
- **Ultralytics YOLO**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

## License

This project is for educational and research purposes.

## Next Steps

1. ✓ Implement hand tracking with YOLO11n
2. ✓ Implement face tracking with MediaPipe
3. ✓ Add emoji reactions and sound effects
4. ✓ Create optimization scripts for Jetson Nano
5. ⏳ Test on actual Jetson Nano hardware
6. ⏳ Fine-tune thresholds for production use
7. ⏳ Add more emoji states and gestures

---

**For detailed optimization instructions, see [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)**
