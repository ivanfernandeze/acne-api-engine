"""Score a trained detector on the frozen test split without retraining.

    uv run python -m training.detection.evaluate_detection --model yolo
    uv run python -m training.detection.evaluate_detection --model fasterrcnn

Thin front door: each trainer already evaluates with ``--skip-train`` through the shared
``common.evaluate_predictions``; this module only picks the right one so the command reads the
same for both. Extra arguments are passed through (``--conf``, ``--tag``, ``--run-dir``…).
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa un detector sobre el test congelado.")
    parser.add_argument("--model", required=True, choices=["yolo", "fasterrcnn"])
    args, passthrough = parser.parse_known_args()

    if args.model == "yolo":
        from training.detection import train_yolo_det as module
    else:
        from training.detection import train_fasterrcnn as module

    sys.argv = [sys.argv[0], "--skip-train", *passthrough]
    module.main()


if __name__ == "__main__":
    main()
