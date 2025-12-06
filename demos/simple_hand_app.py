"""
Simple hand-tracking viewer inspired by simple-mediapipe-project.

- Single-window, minimal HUD
- Keeps pruning/quantization knobs (`--precision`, `--prune`)
- Uses the HandTrackingPipeline for ONNX+TensorRT/CUDA/CPU backends
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_tracking import HandTrackingPipeline, draw_landmarks, draw_detections


def overlay_hud(frame, fps, precision, prune_ratio, hand_count):
    """Overlay a compact HUD with perf/precision info."""
    cv2.putText(frame, f"{fps:.1f} FPS | {precision.upper()} | prune {prune_ratio*100:.0f}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Hands: {hand_count}  |  Quit: q / ESC",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def run(camera=0, precision='fp16', prune=0.0, mirror=True):
    pipeline = HandTrackingPipeline(prune_ratio=prune, precision=precision)
    pipeline.print_stats()

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    fps_hist = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mirror:
            frame = frame[:, ::-1].copy()

        start = time.time()
        landmarks, detections, scale, pad = pipeline.process_frame(frame)
        elapsed = time.time() - start

        fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_hist.append(fps)
        if len(fps_hist) > 30:
            fps_hist.pop(0)
        avg_fps = float(np.mean(fps_hist)) if fps_hist else fps

        vis = frame.copy()
        for lm in landmarks:
            draw_landmarks(vis, lm)
        if len(detections) > 0:
            draw_detections(vis, detections, scale, pad)

        overlay_hud(vis, avg_fps, precision, prune, len(landmarks))
        cv2.imshow("Simple Hand Tracking", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Minimal hand-tracking viewer.")
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16',
                        help='Model precision. Use fp16 for Jetson/TensorRT, int8 for CPU.')
    parser.add_argument('--prune', type=float, default=0.0, help='Pruning ratio (0-1)')
    parser.add_argument('--no-mirror', action='store_true', help='Disable horizontal flip')
    args = parser.parse_args()

    run(camera=args.camera, precision=args.precision, prune=args.prune, mirror=not args.no_mirror)


if __name__ == "__main__":
    main()
