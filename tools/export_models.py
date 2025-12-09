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
    parser = argparse.ArgumentParser(description="Export/prune/quantize hand models from custom weights (deprecated).")
    parser.add_argument('--detector-weights', help='(Deprecated) Path to detector .pth (blazepalm).')
    parser.add_argument('--landmark-weights', help='(Deprecated) Path to landmark .pth (blazehand_landmark).')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp32',
                        help='Export precision (ignored in PyTorch-only runtime).')
    parser.add_argument('--prune', type=float, default=0.0, help='Prune ratio (ignored).')
    parser.add_argument('--prune-mode', choices=['magnitude', 'channel_l1'], default='magnitude',
                        help='Pruning mode (ignored).')
    parser.add_argument('--tag', type=str, default=None, help='Append tag to ONNX filenames (ignored).')
    parser.parse_args()

    raise SystemExit("Export path removed: runtime now loads cached ONNX via onnx2pytorch (no new exports).")


if __name__ == "__main__":
    main()
