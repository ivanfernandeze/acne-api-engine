"""Dataset and stratified splits for the IGA classifier.

Covers: Fase 3 de CRISP-DM (seccion 8.2). Splits are 80/10/10 stratified by IGA class and age
group, and the test split is frozen for the whole thesis: re-running with the same seed must
reproduce exactly the same partition, or no result reported six months apart is comparable.

Expects a manifest CSV with at least: ``path,iga_class`` and optionally ``age,fitzpatrick``.
"""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.iga import IGA_CLASSES, IgaClass


@dataclass(frozen=True)
class Sample:
    """One labelled image."""

    path: Path
    iga_class: IgaClass
    age_group: str
    fitzpatrick: str | None


AGE_GROUPS = ("<18", "18-25", "26-35", ">35", "desconocido")


def age_group_of(age: int | None) -> str:
    if age is None:
        return "desconocido"
    if age < 18:
        return "<18"
    if age <= 25:
        return "18-25"
    if age <= 35:
        return "26-35"
    return ">35"


def load_manifest(manifest_path: Path) -> list[Sample]:
    """Read the dataset manifest, skipping rows whose label is not a valid IGA class."""
    samples: list[Sample] = []
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("iga_class", "").strip()
            if label not in IGA_CLASSES:
                continue
            raw_age = row.get("age", "").strip()
            samples.append(
                Sample(
                    path=Path(row["path"]),
                    iga_class=label,  # type: ignore[arg-type]
                    age_group=age_group_of(int(raw_age) if raw_age.isdigit() else None),
                    fitzpatrick=(row.get("fitzpatrick") or "").strip() or None,
                )
            )
    return samples


def stratified_split(
    samples: list[Sample],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[Sample]]:
    """Split into train/val/test, stratified by (IGA class, age group).

    The assignment is derived from a hash of the file path plus the seed rather than from a
    shuffle. That makes the partition STABLE under changes to the dataset: adding new images later
    does not silently move an old image from test into train, which would quietly invalidate every
    metric already reported in the thesis.
    """
    train_ratio, val_ratio, _ = ratios
    buckets: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for sample in samples:
        buckets[(sample.iga_class, sample.age_group)].append(sample)

    splits: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}

    for stratum in sorted(buckets):
        for sample in sorted(buckets[stratum], key=lambda s: str(s.path)):
            digest = hashlib.sha256(f"{seed}:{sample.path}".encode()).digest()
            position = int.from_bytes(digest[:8], "big") / 2**64

            if position < train_ratio:
                splits["train"].append(sample)
            elif position < train_ratio + val_ratio:
                splits["val"].append(sample)
            else:
                splits["test"].append(sample)

    return splits


def class_distribution(samples: list[Sample]) -> dict[str, int]:
    counts = dict.fromkeys(IGA_CLASSES, 0)
    for sample in samples:
        counts[sample.iga_class] += 1
    return counts


def class_weights(samples: list[Sample]) -> list[float]:
    """Inverse-frequency weights for the cross-entropy loss (seccion 8.1).

    Public acne datasets are heavily imbalanced towards the middle grades; without weighting, the
    model learns to never predict `severo`, which is precisely the class a triage tool must not
    miss.
    """
    counts = class_distribution(samples)
    total = sum(counts.values())
    if total == 0:
        return [1.0] * len(IGA_CLASSES)

    weights: list[float] = []
    for iga_class in IGA_CLASSES:
        count = counts[iga_class]
        weights.append(total / (len(IGA_CLASSES) * count) if count > 0 else 0.0)
    return weights


def build_torch_dataset(
    samples: list[Sample],
    input_size: int,
    train: bool,
    crop_scale: tuple[float, float] | None = None,
) -> Any:
    """Create a torchvision dataset. Imported lazily: the engine runs without torch.

    ``crop_scale`` enables ``RandomResizedCrop`` (area fraction range) for the train split. A
    random crop of 70-100 % of the face still contains lesions of every zone, so the label holds,
    while the model stops seeing the exact same pixels every epoch.
    """
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if train:
        resize: Any = (
            transforms.Resize((input_size, input_size))
            if crop_scale is None
            else transforms.RandomResizedCrop(input_size, scale=crop_scale, ratio=(0.9, 1.1))
        )
        pipeline = transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                # Saturation and hue stay at zero on purpose (seccion 8.3).
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.0, hue=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        pipeline = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    class IgaDataset(Dataset):  # type: ignore[misc]
        def __len__(self) -> int:
            return len(samples)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            sample = samples[index]
            with Image.open(sample.path) as source:
                image = source.convert("RGB")
            return pipeline(image), torch.tensor(IGA_CLASSES.index(sample.iga_class))

    return IgaDataset()


def balanced_sampler(samples: list[Sample], seed: int) -> Any:
    """Sampler that draws every class with equal probability (with replacement).

    Complements the loss weights: weighting only rescales the gradient of the rare `severo`
    images, but the model still sees them a handful of times per epoch. Resampling puts them in
    front of the network as often as the majority classes.
    """
    import torch
    from torch.utils.data import WeightedRandomSampler

    counts = class_distribution(samples)
    per_sample = [1.0 / counts[sample.iga_class] for sample in samples]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        per_sample, num_samples=len(samples), replacement=True, generator=generator
    )
