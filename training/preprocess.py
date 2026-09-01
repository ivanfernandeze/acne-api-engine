"""Dataset preparation: face detection, crop and manifest.

Covers: Fase 3 de CRISP-DM (seccion 8.3).

    uv run python -m training.preprocess --source ./data/raw --output ./data/prepared

Expects ``data/raw/<iga_class>/*.jpg`` and writes cropped images plus ``manifest.csv``. The
manifest is what ``training.dataset`` reads, so the split is computed from prepared data, never
from the raw folder.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from app.domain.iga import IGA_CLASSES

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara el dataset para el entrenamiento.")
    parser.add_argument("--source", default="./data/raw")
    parser.add_argument("--output", default="./data/prepared")
    parser.add_argument("--size", type=int, default=260)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument(
        "--detector",
        default="mtcnn",
        choices=["mtcnn", "none"],
        help="Detección facial. Con 'none' se usa un recorte cuadrado centrado.",
    )
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError as error:
        raise SystemExit("Falta Pillow: uv pip install -e '.[dev]'") from error

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    if not source_dir.is_dir():
        raise SystemExit(
            f"No existe {source_dir}. Coloque las imágenes como data/raw/<clase_iga>/*.jpg.\n"
            f"Clases válidas: {', '.join(IGA_CLASSES)}"
        )

    detector = _load_detector(args.detector)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    skipped: list[str] = []
    stats: dict[str, int] = dict.fromkeys(IGA_CLASSES, 0)

    for iga_class in IGA_CLASSES:
        class_dir = source_dir / iga_class
        if not class_dir.is_dir():
            print(f"· sin carpeta para la clase '{iga_class}', se omite")
            continue

        target_dir = output_dir / iga_class
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                cropped = _crop(image, detector, args.margin)
                resized = cropped.resize((args.size, args.size), Image.Resampling.BILINEAR)

                target_path = target_dir / f"{image_path.stem}.png"
                resized.save(target_path, format="PNG", optimize=True)

                rows.append({"path": str(target_path), "iga_class": iga_class, "age": "", "fitzpatrick": ""})
                stats[iga_class] += 1
            except Exception as error:  # noqa: BLE001 - one bad file must not stop the batch
                skipped.append(f"{image_path}: {error}")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "iga_class", "age", "fitzpatrick"])
        writer.writeheader()
        writer.writerows(rows)

    # The class distribution belongs in 02_data_understanding.md: a heavily imbalanced dataset is
    # the single most common reason an acne grader never predicts the severe class.
    (output_dir / "distribution.json").write_text(
        json.dumps({"per_class": stats, "total": len(rows), "skipped": len(skipped)}, indent=2),
        encoding="utf-8",
    )

    print(f"\nPreparadas {len(rows)} imágenes en {output_dir}")
    for iga_class, count in stats.items():
        share = 100 * count / len(rows) if rows else 0
        print(f"  {iga_class:<20} {count:>5}  ({share:5.1f} %)")
    if skipped:
        print(f"\nOmitidas {len(skipped)} imágenes:")
        for entry in skipped[:10]:
            print(f"  - {entry}")

    print(f"\nManifiesto: {manifest_path}")
    print("Siguiente paso:  uv run python -m training.train")


def _load_detector(kind: str) -> Any:
    if kind == "none":
        return None
    try:
        from facenet_pytorch import MTCNN  # type: ignore[import-not-found]

        return MTCNN(keep_all=False)
    except ImportError:
        print("· facenet-pytorch no disponible: se usará recorte cuadrado centrado.")
        return None


def _crop(image: Any, detector: Any, margin: float) -> Any:
    """Crop to the face, falling back to a centred square.

    Falling back rather than discarding keeps hard-to-detect photographs in the dataset; dropping
    them would bias the sample towards easy, well-lit frontal images and inflate every metric.
    """
    if detector is not None:
        try:
            boxes, _ = detector.detect(image)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = (int(v) for v in boxes[0])
                dx = int((x2 - x1) * margin)
                dy = int((y2 - y1) * margin)
                return image.crop(
                    (
                        max(0, x1 - dx),
                        max(0, y1 - dy),
                        min(image.width, x2 + dx),
                        min(image.height, y2 + dy),
                    )
                )
        except Exception:  # noqa: BLE001 - detection is best-effort
            pass

    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


if __name__ == "__main__":
    main()
