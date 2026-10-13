"""Head-pose probe: how the screener's yaw ratio is distributed over a folder of photographs.

Covers: validacion de tomas (vista general frente a perfiles) del cribado de admision.

    uv run python -m training.calibrate_pose --images archive/acne_1024/all_1024 --sample 400

Runs the same detector and the same ``estimate_pose`` as ``LocalScreener`` and prints a histogram
of the yaw ratio and of the decided pose, so ``FRONTAL_MAX_YAW``, ``TURNED_MIN_YAW`` and
``POSE_MIN_SCORE`` can be checked against real photographs. With ``--images`` pointing at a folder
of shots taken for ONE view (for instance only "Vista general"), the share that lands in the wrong
pose is the rate at which that view would be refused.

Reference run (400 random ACNE04 photographs, 01/10/2026): no face missed; |yaw| < 0,03 for the
few frontal faces, 0,06 for a head turned ~30°, up to 0,30 for profiles; cheek close-ups come with
detection scores 0,30-0,51 and are left ``indeterminado``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from app.models.local_screener import (
    MIN_DETECTION_FRACTION,
    LocalScreener,
    _merge_overlapping,
    estimate_pose,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
YAW_BINS = [
    -1, -0.3, -0.2, -0.15, -0.12, -0.09, -0.06, -0.04, 0,
    0.04, 0.06, 0.09, 0.12, 0.15, 0.2, 0.3, 1,
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribución de la pose de cabeza del cribado.")
    parser.add_argument("--images", required=True, help="Carpeta con fotografías.")
    parser.add_argument("--sample", type=int, default=0, help="Muestra aleatoria (0 = todas).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="JSON opcional con la pose de cada imagen.")
    args = parser.parse_args()

    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.sample and args.sample < len(files):
        files = random.Random(args.seed).sample(files, args.sample)

    screener = LocalScreener()
    cv2 = screener._cv2
    rows: list[dict[str, object]] = []
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            rows.append({"file": path.name, "pose": "ilegible", "yaw": None, "score": None})
            continue
        faces, analysed = screener._detect(image)
        plausible = [
            f
            for f in _merge_overlapping(faces)
            if f.width >= MIN_DETECTION_FRACTION * analysed.shape[1]
            and f.height >= MIN_DETECTION_FRACTION * analysed.shape[0]
        ]
        best = max(plausible, key=lambda f: f.score, default=None)
        if best is None:
            rows.append({"file": path.name, "pose": "sin_rostro", "yaw": None, "score": None})
            continue
        pose, yaw = estimate_pose(best)
        rows.append({"file": path.name, "pose": pose, "yaw": yaw, "score": round(best.score, 2)})

    print(f"Imágenes: {len(rows)}")
    for pose, count in Counter(str(r["pose"]) for r in rows).most_common():
        print(f"  {pose:<16} {count:>5}  ({count / max(1, len(rows)):.1%})")
    yaws = np.array([r["yaw"] for r in rows if r["yaw"] is not None], dtype=float)
    if yaws.size:
        counts, edges = np.histogram(yaws, bins=YAW_BINS)
        print("\nyaw_ratio:")
        for count, low, high in zip(counts, edges[:-1], edges[1:], strict=True):
            print(f"  [{low:+.2f}, {high:+.2f})  {count:>5}")

    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nDetalle en {args.output}")


if __name__ == "__main__":
    main()
