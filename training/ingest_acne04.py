"""Ingest ACNE04 into the ``data/raw/<iga_class>/`` layout that ``training.preprocess`` expects.

Covers: Fase 2-3 de CRISP-DM (seccion 8.2, "ACNE04 para pre-entrenamiento especifico").

    uv run python -m training.ingest_acne04 --source ./archive/acne_1024/all_1024

ACNE04 (Wu et al., 2019) grades each face with the four-level Hayashi scale, encoded in the file
name prefix ``levle<N>_<id>.jpg`` (the ``levle`` typo is the dataset's own). Hayashi counts
papules/pustules: 0 = mild (0-5), 1 = moderate (6-20), 2 = severe (21-50), 3 = very severe (>50).

The engine is wired to the five IGA grades (``app/domain/iga.py``). The mapping below keeps the four
Hayashi levels separable instead of collapsing two of them: Hayashi "mild" images have at most five
lesions, which is what IGA 1 (casi sin lesiones) describes, and the remaining three levels shift up
one grade. ``sin_lesiones`` (IGA 0) receives no images: ACNE04 contains no clear skin, so that class
stays at zero support (and zero loss weight) until a dataset with healthy faces is added.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from app.domain.iga import IGA_CLASSES, IgaClass

HAYASHI_TO_IGA: dict[int, IgaClass] = {
    0: "casi_sin_lesiones",
    1: "leve",
    2: "moderado",
    3: "severo",
}

FILENAME_PATTERN = re.compile(r"^levle(\d)_(\d+)\.(jpe?g|png|webp)$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convierte ACNE04 al layout data/raw/<clase>/.")
    parser.add_argument("--source", default="./archive/acne_1024/all_1024")
    parser.add_argument("--output", default="./data/raw")
    parser.add_argument(
        "--link",
        action="store_true",
        help="Crear hard links en lugar de copiar (mismo volumen). Por defecto se copia.",
    )
    args = parser.parse_args()

    for iga_class in HAYASHI_TO_IGA.values():
        if iga_class not in IGA_CLASSES:
            raise SystemExit(f"Mapeo inválido: '{iga_class}' no está en IGA_CLASSES.")

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    if not source_dir.is_dir():
        raise SystemExit(f"No existe {source_dir}. Descomprima ACNE04 en archive/acne_1024/.")

    stats: dict[str, int] = dict.fromkeys(IGA_CLASSES, 0)
    ignored: list[str] = []

    for image_path in sorted(source_dir.iterdir()):
        if not image_path.is_file():
            continue
        match = FILENAME_PATTERN.match(image_path.name)
        if match is None:
            ignored.append(image_path.name)
            continue

        level = int(match.group(1))
        iga_class = HAYASHI_TO_IGA.get(level)
        if iga_class is None:
            ignored.append(f"{image_path.name} (nivel Hayashi {level} desconocido)")
            continue

        target_dir = output_dir / iga_class
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / image_path.name
        if target_path.exists():
            target_path.unlink()
        if args.link:
            target_path.hardlink_to(image_path)
        else:
            shutil.copy2(image_path, target_path)
        stats[iga_class] += 1

    total = sum(stats.values())
    print(f"\nIngeridas {total} imágenes de {source_dir} en {output_dir}")
    for iga_class, count in stats.items():
        share = 100 * count / total if total else 0
        print(f"  {iga_class:<20} {count:>5}  ({share:5.1f} %)")

    # Anything that is not a labelled image (caption .jsonl, helper scripts) is expected here; a
    # long list means the source folder is not the one we think it is.
    if ignored:
        print(f"\nIgnorados {len(ignored)} archivos sin etiqueta Hayashi en el nombre:")
        for entry in ignored[:10]:
            print(f"  - {entry}")

    print("\nSiguiente paso:  uv run python -m training.preprocess --detector none")


if __name__ == "__main__":
    main()
