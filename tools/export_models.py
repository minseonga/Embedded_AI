"""
Export/prune/quantize hand models from custom weights.

Examples:
  python tools/export_models.py \
      --detector-weights /path/to/blazepalm.pth \
      --landmark-weights /path/to/blazehand_landmark.pth \
      --precision fp16 --prune 0.3 --tag custom

Artifacts are written to assets/models with names including the tag.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hand_tracking import HandTrackingPipeline  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Export/prune/quantize hand models from custom weights.")
    parser.add_argument('--detector-weights', required=True, help='Path to detector .pth (blazepalm).')
    parser.add_argument('--landmark-weights', required=True, help='Path to landmark .pth (blazehand_landmark).')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp32',
                        help='Export precision (fp16/ int8).')
    parser.add_argument('--prune', type=float, default=0.0, help='Prune ratio (0-1).')
    parser.add_argument('--prune-mode', choices=['magnitude', 'channel_l1'], default='magnitude',
                        help='Pruning mode: unstructured magnitude or L1 channel pruning.')
    parser.add_argument('--tag', type=str, default=None, help='Append tag to ONNX filenames.')
    args = parser.parse_args()

    pipeline = HandTrackingPipeline(
        detector_weights=args.detector_weights,
        landmark_weights=args.landmark_weights,
        precision=args.precision,
        prune_ratio=args.prune,
        prune_mode=args.prune_mode,
        model_tag=args.tag,
        build_only=True,  # just export/cache
    )

    print("Export complete.")
    print(f"Detector ONNX: assets/models/{pipeline._get_detector_name()}")
    print(f"Landmark ONNX: assets/models/{pipeline._get_landmark_name()}")


if __name__ == "__main__":
    main()
