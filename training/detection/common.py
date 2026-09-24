"""Shared pieces of the detection pipeline: dataset reader, count → grade, one evaluator.

Both detectors (YOLOv8 and the Faster R-CNN baseline) are scored by ``evaluate_predictions`` on
the same held-out images, so the comparison in ``05_evaluation.md`` measures the architecture and
nothing else. torchmetrics' ``MeanAveragePrecision`` is the COCO evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.domain.iga import IGA_CLASSES, IgaClass

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

#: Hayashi (2008) severity from the lesion count, expressed on the system's IGA vocabulary with
#: the same mapping the classifier's data uses (02_data_understanding.md §3). This is what makes
#: the detector a second, independent reading of severity.
HAYASHI_BANDS: tuple[tuple[int, IgaClass], ...] = (
    (5, "casi_sin_lesiones"),
    (20, "leve"),
    (50, "moderado"),
)


def iga_from_count(count: int) -> IgaClass:
    for upper, iga_class in HAYASHI_BANDS:
        if count <= upper:
            return iga_class
    return "severo"


@dataclass(frozen=True)
class DetectionSample:
    """One image of a YOLO-layout split with its ground-truth boxes (absolute xyxy)."""

    image: Path
    boxes: list[tuple[float, float, float, float]]
    width: int
    height: int


def read_split(root: Path, split: str) -> list[DetectionSample]:
    """Read ``<root>/images/<split>`` + ``<root>/labels/<split>`` (YOLO normalised format)."""
    from PIL import Image

    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    samples: list[DetectionSample] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        with Image.open(image_path) as source:
            width, height = source.size
        boxes: list[tuple[float, float, float, float]] = []
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, w, h = (float(v) for v in parts)
                boxes.append(
                    (
                        (cx - w / 2) * width,
                        (cy - h / 2) * height,
                        (cx + w / 2) * width,
                        (cy + h / 2) * height,
                    )
                )
        samples.append(DetectionSample(image_path, boxes, width, height))
    return samples


@dataclass
class Prediction:
    """Detector output for one image: absolute xyxy boxes and their confidences."""

    boxes: list[tuple[float, float, float, float]]
    scores: list[float]


def evaluate_predictions(
    samples: list[DetectionSample], predictions: list[Prediction], conf_threshold: float
) -> dict[str, Any]:
    """COCO mAP over all confidences, plus the counting metrics at ``conf_threshold``.

    Counting is what the clinic uses, so it is scored at the operating point the engine will run
    with: boxes under the threshold do not exist for the count, the MAE, or the Hayashi grade.
    """
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from torchmetrics.detection import MeanAveragePrecision

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    counts_true: list[int] = []
    counts_pred: list[int] = []
    grade_true: list[int] = []
    grade_pred: list[int] = []
    tp = fp = fn = 0

    for sample, prediction in zip(samples, predictions, strict=True):
        kept = [
            (b, s)
            for b, s in zip(prediction.boxes, prediction.scores, strict=True)
            if s >= conf_threshold
        ]
        metric.update(
            [
                {
                    "boxes": torch.tensor(prediction.boxes, dtype=torch.float32).reshape(-1, 4),
                    "scores": torch.tensor(prediction.scores, dtype=torch.float32),
                    "labels": torch.zeros(len(prediction.boxes), dtype=torch.int64),
                }
            ],
            [
                {
                    "boxes": torch.tensor(sample.boxes, dtype=torch.float32).reshape(-1, 4),
                    "labels": torch.zeros(len(sample.boxes), dtype=torch.int64),
                }
            ],
        )
        matched = _match(sample.boxes, [b for b, _ in kept], iou_threshold=0.5)
        tp += matched
        fp += len(kept) - matched
        fn += len(sample.boxes) - matched
        counts_true.append(len(sample.boxes))
        counts_pred.append(len(kept))
        grade_true.append(IGA_CLASSES.index(iga_from_count(len(sample.boxes))))
        grade_pred.append(IGA_CLASSES.index(iga_from_count(len(kept))))

    summary = {k: float(v) for k, v in metric.compute().items() if hasattr(v, "item")}
    true = np.array(counts_true)
    pred = np.array(counts_pred)
    correlation = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 and true.std() > 0 else None
    grades_true = np.array(grade_true)
    grades_pred = np.array(grade_pred)

    return {
        "n": len(samples),
        "conf_threshold": conf_threshold,
        "map50": round(summary.get("map_50", 0.0), 6),
        "map50_95": round(summary.get("map", 0.0), 6),
        "precision_at_threshold": round(tp / (tp + fp), 6) if tp + fp else 0.0,
        "recall_at_threshold": round(tp / (tp + fn), 6) if tp + fn else 0.0,
        "count_mae": round(float(np.abs(true - pred).mean()), 6) if len(true) else 0.0,
        "count_pearson_r": None if correlation is None else round(correlation, 6),
        "count_grade_accuracy": round(float((grades_true == grades_pred).mean()), 6)
        if len(true)
        else 0.0,
        "count_grade_within_one": round(float((np.abs(grades_true - grades_pred) <= 1).mean()), 6)
        if len(true)
        else 0.0,
        "count_grade_f1_macro": round(
            float(f1_score(grade_true, grade_pred, average="macro", zero_division=0)), 6
        )
        if len(true)
        else 0.0,
        "gt_boxes": int(true.sum()),
        "pred_boxes": int(pred.sum()),
    }


def _match(
    truth: list[tuple[float, float, float, float]],
    predicted: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> int:
    """Greedy one-to-one matching by IoU; returns the number of true positives."""
    remaining = list(truth)
    matched = 0
    for box in predicted:
        best_index = -1
        best_iou = iou_threshold
        for index, gt in enumerate(remaining):
            iou = _iou(box, gt)
            if iou >= best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0:
            remaining.pop(best_index)
            matched += 1
    return matched


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
