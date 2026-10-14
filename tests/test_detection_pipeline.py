"""The detection pipeline works end to end on a synthetic Pascal VOC set.

No ACNE04 Detection data is on disk yet, so the ingest, the shared evaluator and the count → grade
rule are proven on twelve generated images with generated boxes. The two 1-epoch training smokes
(YOLOv8 from a yaml, Faster R-CNN without pretrained weights) run only with
``ACNE_SMOKE_DETECTORS=1``: they take a minute on CPU and are not part of the default suite.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

pytest.importorskip("torch")

from training.detection.common import (
    DetectionSample,
    Prediction,
    evaluate_predictions,
    iga_from_count,
    read_split,
)

ENGINE_ROOT = Path(__file__).resolve().parents[1]


def _write_voc_fixture(root: Path, count: int = 12) -> None:
    images = root / "JPEGImages"
    annotations = root / "Annotations"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)
    for index in range(count):
        level = index % 4
        stem = f"levle{level}_{index}"
        image = Image.new("RGB", (96, 80), (200, 170, 150))
        draw = ImageDraw.Draw(image)
        boxes = []
        for k in range(level + 1):
            x1, y1 = 8 + 20 * k, 10 + 12 * k
            boxes.append((x1, y1, x1 + 12, y1 + 12))
            draw.ellipse((x1, y1, x1 + 12, y1 + 12), fill=(180, 60, 60))
        image.save(images / f"{stem}.jpg", quality=90)
        objects = "".join(
            f"<object><name>acne</name><bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin>"
            f"<xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"
            for x1, y1, x2, y2 in boxes
        )
        (annotations / f"{stem}.xml").write_text(
            f"<annotation><filename>{stem}.jpg</filename>"
            f"<size><width>96</width><height>80</height><depth>3</depth></size>{objects}</annotation>",
            encoding="utf-8",
        )


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "ACNE04_Detection"
    _write_voc_fixture(source)
    config = {
        "seed": 42,
        "dataset_version": "vtest",
        "data_dir": str(tmp_path / "no-manifest"),
        "output_dir": str(tmp_path / "checkpoints"),
        "tracking_dir": str(tmp_path / "runs"),
        "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return {"source": source, "config": config_path, "output": tmp_path / "detection"}


def _run(module: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ENGINE_ROOT,
        capture_output=True,
        text=True,
        check=False,
        # Windows consoles default to cp1252; the scripts print Spanish and symbols.
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        encoding="utf-8",
    )


def test_ingest_converts_voc_to_yolo_with_a_hash_split(workspace: dict[str, Path]) -> None:
    result = _run(
        "training.detection.ingest_acne04_detection",
        "--config",
        str(workspace["config"]),
        "--source",
        str(workspace["source"]),
        "--output",
        str(workspace["output"]),
        "--copy",
    )
    assert result.returncode == 0, result.stderr

    output = workspace["output"]
    data = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == {0: "acne"}
    distribution = json.loads((output / "distribution.json").read_text(encoding="utf-8"))
    total = sum(distribution[s]["images"] for s in ("train", "val", "test"))
    assert total == 12
    assert distribution["skipped"] == []

    samples = [s for split in ("train", "val", "test") for s in read_split(output, split)]
    assert len(samples) == 12
    # levle3_* carries four boxes, levle0_* one: the XML boxes survived the conversion.
    by_level = {int(s.image.stem[5]): len(s.boxes) for s in samples}
    assert by_level == {0: 1, 1: 2, 2: 3, 3: 4}
    # Every box is inside its image and back in pixel coordinates.
    for sample in samples:
        for x1, y1, x2, y2 in sample.boxes:
            assert 0 <= x1 < x2 <= sample.width and 0 <= y1 < y2 <= sample.height


def test_count_to_grade_follows_hayashi_bands() -> None:
    assert iga_from_count(0) == "casi_sin_lesiones"
    assert iga_from_count(5) == "casi_sin_lesiones"
    assert iga_from_count(6) == "leve"
    assert iga_from_count(20) == "leve"
    assert iga_from_count(21) == "moderado"
    assert iga_from_count(50) == "moderado"
    assert iga_from_count(51) == "severo"


def test_evaluator_scores_perfect_and_empty_predictions() -> None:
    samples = [
        DetectionSample(Path("a.jpg"), [(0, 0, 10, 10), (20, 20, 30, 30)], 100, 100),
        DetectionSample(Path("b.jpg"), [(5, 5, 15, 15)], 100, 100),
    ]
    perfect = [
        Prediction([(0, 0, 10, 10), (20, 20, 30, 30)], [0.9, 0.8]),
        Prediction([(5, 5, 15, 15)], [0.95]),
    ]
    metrics = evaluate_predictions(samples, perfect, conf_threshold=0.5)
    assert metrics["map50"] == pytest.approx(1.0)
    assert metrics["count_mae"] == 0.0
    assert metrics["count_grade_accuracy"] == 1.0
    assert metrics["recall_at_threshold"] == 1.0

    nothing = [Prediction([], []), Prediction([], [])]
    metrics = evaluate_predictions(samples, nothing, conf_threshold=0.5)
    assert metrics["map50"] == pytest.approx(0.0)
    assert metrics["count_mae"] == pytest.approx(1.5)
    assert metrics["recall_at_threshold"] == 0.0


@pytest.mark.skipif(
    os.environ.get("ACNE_SMOKE_DETECTORS") != "1", reason="ACNE_SMOKE_DETECTORS=1 to run"
)
def test_detector_trainers_complete_one_epoch(workspace: dict[str, Path]) -> None:
    ingest = _run(
        "training.detection.ingest_acne04_detection",
        "--config",
        str(workspace["config"]),
        "--source",
        str(workspace["source"]),
        "--output",
        str(workspace["output"]),
        "--copy",
    )
    assert ingest.returncode == 0, ingest.stderr

    yolo = _run(
        "training.detection.train_yolo_det",
        "--config",
        str(workspace["config"]),
        "--data",
        str(workspace["output"]),
        "--weights",
        "yolov8n.yaml",
        "--imgsz",
        "64",
        "--epochs",
        "1",
        "--batch",
        "4",
        "--device",
        "cpu",
    )
    assert yolo.returncode == 0, yolo.stderr[-2000:]
    assert "mAP50" in yolo.stdout

    frcnn = _run(
        "training.detection.train_fasterrcnn",
        "--config",
        str(workspace["config"]),
        "--data",
        str(workspace["output"]),
        "--epochs",
        "1",
        "--batch",
        "2",
        "--imgsz",
        "96",
        "--no-pretrained",
    )
    assert frcnn.returncode == 0, frcnn.stderr[-2000:]
    assert "mAP50" in frcnn.stdout
