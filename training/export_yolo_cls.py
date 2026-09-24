"""Materialise the frozen split as the folder layout Ultralytics classification expects.

Covers: Fase 4 de CRISP-DM, seleccion de modelo.

    uv run python -m training.export_yolo_cls --config training/config.yaml

Writes ``data/yolo_cls/{train,val,test}/<iga_class>/<stem>.png`` as HARD LINKS to the prepared
images, so the 170 MB of ``data/prepared`` are not duplicated and every model — timm or YOLO —
trains on byte-identical files. The partition is the one ``training.dataset.stratified_split``
derives from the manifest, which is what keeps the test set frozen across architectures: a
comparison on different test images is not a comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from training.dataset import class_distribution, load_manifest, stratified_split

SPLITS = ("train", "val", "test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta el split congelado en carpetas YOLO-cls.")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default="./data/yolo_cls")
    parser.add_argument("--copy", action="store_true", help="Copiar en vez de crear hard links.")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = Path(args.manifest or Path(config["data_dir"]) / "manifest.csv")
    samples = load_manifest(manifest)
    splits = stratified_split(
        samples,
        (config["splits"]["train"], config["splits"]["val"], config["splits"]["test"]),
        int(config["seed"]),
    )

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)

    summary: dict[str, Any] = {}
    for split in SPLITS:
        # Only classes with images get a folder: torchvision's ImageFolder, which Ultralytics
        # wraps, refuses an empty class directory. Ultralytics then numbers the folders it finds
        # alphabetically; `train_yolo_cls` maps those names back to IGA_CLASSES positions.
        for sample in splits[split]:
            target = output / split / sample.iga_class / sample.path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.copy:
                shutil.copy2(sample.path, target)
            else:
                os.link(sample.path, target)
        summary[split] = {"total": len(splits[split]), **class_distribution(splits[split])}

    (output / "split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Split exportado en {output} ({'copias' if args.copy else 'hard links'})")
    for split in SPLITS:
        print(f"  {split:<6} {summary[split]['total']:>5}")
    print("Siguiente paso:  uv run python -m training.train_yolo_cls")


if __name__ == "__main__":
    main()
