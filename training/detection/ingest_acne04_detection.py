"""ACNE04 *Detection* (Pascal VOC) → YOLO layout, with the SAME frozen split as the classifier.

Covers: Fase 2-3 de CRISP-DM for the lesion detector.

    uv run python -m training.detection.ingest_acne04_detection \\
        --source ./archive/ACNE04_Detection --output ./data/detection

Expected source (the ``Detection`` folder of ACNE04, Wu et al. 2019, ICCV — released through
``github.com/xpwu95/ldl``; also redistributed on Kaggle/Roboflow):

    <source>/JPEGImages/levle<N>_<id>.jpg      original resolution
    <source>/Annotations/levle<N>_<id>.xml     Pascal VOC, one <object> per lesion, class "acne"

Output:

    data/detection/images/{train,val,test}/<stem>.jpg   (hard links)
    data/detection/labels/{train,val,test}/<stem>.txt   "0 cx cy w h" normalised, one line per box
    data/detection/data.yaml                            Ultralytics dataset file
    data/detection/distribution.json                    images, boxes, boxes/image per split

**Split alignment.** Every stem is assigned to the split the classifier's manifest put it in
(``training.dataset.stratified_split`` over ``data/prepared/manifest.csv``, same seed and ratios):
the 142 test faces the severity model never saw are the same 142 the detector never sees, so the
two components can be cross-checked image by image. Stems absent from the manifest (ACNE04 has
1457 images, the local classification copy 1406) go to train.

Boxes stay in the original image coordinates; Ultralytics and torchvision resize at load time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from training.dataset import load_manifest, stratified_split

STEM_PATTERN = re.compile(r"^levle(\d)_(\d+)$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte ACNE04 Detection (VOC) a YOLO.")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--source", default="./archive/ACNE04_Detection")
    parser.add_argument("--output", default="./data/detection")
    parser.add_argument("--copy", action="store_true", help="Copiar en vez de crear hard links.")
    args = parser.parse_args()

    source = Path(args.source)
    images_dir = _first_existing(source, ("JPEGImages", "images", "Images"))
    labels_dir = _first_existing(source, ("Annotations", "annotations", "xml"))
    if images_dir is None or labels_dir is None:
        raise SystemExit(
            f"No se encontró {source}/JPEGImages y {source}/Annotations.\n"
            "Descargue la mitad Detection de ACNE04 (Wu et al., 2019 — github.com/xpwu95/ldl, "
            "enlace a Google Drive; también en Kaggle/Roboflow) y descomprímala en "
            f"{source}/ con las carpetas JPEGImages/ y Annotations/."
        )

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    split_of_stem = _classifier_split(config)

    output = Path(args.output)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    unlisted: list[str] = []
    skipped: list[str] = []

    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = image_path.stem
        if STEM_PATTERN.match(stem) is None:
            skipped.append(f"{image_path.name}: nombre fuera del patrón levle<N>_<id>")
            continue
        xml_path = labels_dir / f"{stem}.xml"
        if not xml_path.is_file():
            skipped.append(f"{image_path.name}: sin {xml_path.name}")
            continue

        split = split_of_stem.get(stem)
        if split is None:
            # Not in the classifier's manifest: with a manifest these go to train (they can never
            # leak into the shared test); without one, the same hash rule as the classifier.
            split = "train" if split_of_stem else _hash_split(stem, config)
            unlisted.append(stem)

        boxes = _read_voc(xml_path)
        target_image = output / "images" / split / image_path.name
        target_label = output / "labels" / split / f"{stem}.txt"
        if target_image.exists():
            target_image.unlink()
        if args.copy:
            import shutil

            shutil.copy2(image_path, target_image)
        else:
            os.link(image_path, target_image)
        target_label.write_text(
            "".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes)
        )

        stats[split]["images"] += 1
        stats[split]["boxes"] += len(boxes)

    data_yaml = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "acne"},
    }
    (output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    distribution = {
        split: {
            "images": stats[split]["images"],
            "boxes": stats[split]["boxes"],
            "boxes_per_image": round(stats[split]["boxes"] / stats[split]["images"], 2)
            if stats[split]["images"]
            else 0.0,
        }
        for split in SPLITS
    }
    distribution["unlisted_in_classifier_manifest"] = len(unlisted)
    distribution["skipped"] = skipped
    (output / "distribution.json").write_text(json.dumps(distribution, indent=2), encoding="utf-8")

    print(f"Dataset de detección en {output}")
    for split in SPLITS:
        d = distribution[split]
        print(
            f"  {split:<6} {d['images']:>5} imágenes  {d['boxes']:>6} cajas  "
            f"({d['boxes_per_image']} por imagen)"
        )
    if unlisted:
        print(
            f"  {len(unlisted)} imágenes no estaban en el manifiesto del clasificador; van a train"
        )
    if skipped:
        print(f"\nOmitidas {len(skipped)}:")
        for entry in skipped[:10]:
            print(f"  - {entry}")
    print("\nSiguiente paso:  uv run python -m training.detection.train_yolo_det")


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _classifier_split(config: dict[str, Any]) -> dict[str, str]:
    """stem → split, as the severity classifier's manifest fixes it. Empty if no manifest yet."""
    manifest = Path(config["data_dir"]) / "manifest.csv"
    if not manifest.is_file():
        print("· sin manifest.csv del clasificador: split por hash del nombre de archivo")
        return {}
    samples = load_manifest(manifest)
    splits = stratified_split(
        samples,
        (config["splits"]["train"], config["splits"]["val"], config["splits"]["test"]),
        int(config["seed"]),
    )
    return {sample.path.stem: split for split in SPLITS for sample in splits[split]}


def _hash_split(stem: str, config: dict[str, Any]) -> str:
    """The classifier's rule (dataset.stratified_split) applied to the stem alone."""
    import hashlib

    digest = hashlib.sha256(f"{int(config['seed'])}:{stem}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < float(config["splits"]["train"]):
        return "train"
    if position < float(config["splits"]["train"]) + float(config["splits"]["val"]):
        return "val"
    return "test"


def _read_voc(xml_path: Path) -> list[tuple[float, float, float, float]]:
    """Pascal VOC boxes → normalised YOLO (cx, cy, w, h), clipped to the image."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width")) if size is not None else 0.0
    height = float(size.findtext("height")) if size is not None else 0.0
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_path.name}: <size> inválido")

    boxes: list[tuple[float, float, float, float]] = []
    for obj in root.iter("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        x1 = max(0.0, float(bndbox.findtext("xmin")))
        y1 = max(0.0, float(bndbox.findtext("ymin")))
        x2 = min(width, float(bndbox.findtext("xmax")))
        y2 = min(height, float(bndbox.findtext("ymax")))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            (
                (x1 + x2) / 2 / width,
                (y1 + y2) / 2 / height,
                (x2 - x1) / width,
                (y2 - y1) / height,
            )
        )
    return boxes


if __name__ == "__main__":
    main()
