"""Faster R-CNN ResNet-50 FPN — the detection baseline the thesis compares YOLOv8 against.

Covers: Fase 4-5 de CRISP-DM for the detector (linea base).

    uv run python -m training.detection.train_fasterrcnn --config training/config.yaml

torchvision's ``fasterrcnn_resnet50_fpn_v2`` with COCO weights, box predictor replaced by two
classes (background + acne). Trained with SGD on the same ``data/detection`` split, selected by
validation mAP50, and scored on the frozen test split by the same ``evaluate_predictions`` as
YOLOv8, so the two detectors are compared by one evaluator.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.detection.common import DetectionSample, Prediction, evaluate_predictions, read_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena Faster R-CNN ResNet-50 (línea base).")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--data", default="./data/detection")
    parser.add_argument("--tag", default="fasterrcnn-r50")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=640, help="Lado menor al que se reescala")
    parser.add_argument("--conf", type=float, default=0.5, help="Umbral de conteo")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = Path(args.data)
    if not (data / "images" / "train").is_dir():
        raise SystemExit(
            f"No existe {data}. Ejecute antes: python -m training.detection.ingest_acne04_detection"
        )
    tracking = Path(config["tracking_dir"]) / config["dataset_version"] / "detection"
    tracking.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(config["output_dir"]) / f"{args.tag.replace('-', '_')}_lesiones.pt"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    weights = "DEFAULT" if args.pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=args.imgsz, max_size=args.imgsz)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    model.to(device)

    train_samples = read_split(data, "train")
    val_samples = read_split(data, "val")
    test_samples = read_split(data, "test")

    if not args.skip_train:
        history = _fit(
            model, train_samples, val_samples, device, args, seed, checkpoint_path, run_id, tracking
        )
    else:
        history = []
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["state_dict"])

    model.eval()
    predictions, latencies = _predict(model, test_samples, device)
    payload = evaluate_predictions(test_samples, predictions, conf_threshold=args.conf)
    payload.update(
        {
            "model": "fasterrcnn_resnet50_fpn_v2",
            "tag": args.tag,
            "imgsz": args.imgsz,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "dataset_version": config["dataset_version"],
            "split": "test",
            "model_version": f"{args.tag}_{run_id}",
            "seed": seed,
            "epochs_run": len(history),
            "inference_ms_median": round(sorted(latencies[1:])[len(latencies[1:]) // 2], 2)
            if len(latencies) > 2
            else None,
        }
    )
    output = tracking / f"metrics-detection-{args.tag}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nDetector '{args.tag}' · test · n = {payload['n']}")
    print(
        f"  mAP50 {payload['map50']:.4f} · mAP50-95 {payload['map50_95']:.4f} · "
        f"MAE conteo {payload['count_mae']:.2f}"
    )
    print(f"Métricas escritas en {output}")


def _fit(
    model: Any,
    train_samples: list[DetectionSample],
    val_samples: list[DetectionSample],
    device: Any,
    args: argparse.Namespace,
    seed: int,
    checkpoint_path: Path,
    run_id: str,
    tracking: Path,
) -> list[dict[str, float]]:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(
        _TorchDataset(train_samples, train=True),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: tuple(zip(*batch, strict=True)),
        generator=torch.Generator().manual_seed(seed),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, float]] = []
    best_map = -1.0
    run_dir = tracking / f"{args.tag}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        total = 0.0
        batches = 0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        scheduler.step()

        model.eval()
        predictions, _ = _predict(model, val_samples, device)
        val = evaluate_predictions(val_samples, predictions, conf_threshold=args.conf)
        record = {
            "epoch": epoch,
            "train_loss": round(total / max(batches, 1), 6),
            "val_map50": val["map50"],
            "val_map50_95": val["map50_95"],
            "val_count_mae": val["count_mae"],
            "seconds": round(time.perf_counter() - started, 2),
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{args.epochs}  loss={record['train_loss']:.4f}  "
            f"val_mAP50={val['map50']:.4f}  val_MAE={val['count_mae']:.2f}"
        )
        if val["map50"] > best_map:
            best_map = val["map50"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_name": "fasterrcnn_resnet50_fpn_v2",
                    "version": f"{args.tag}_{run_id}",
                    "seed": seed,
                    "epoch": epoch,
                    "val_map50": best_map,
                    "imgsz": args.imgsz,
                },
                checkpoint_path,
            )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"\nMejor mAP50 de validación: {best_map:.4f} · checkpoint {checkpoint_path}")
    return history


def _predict(
    model: Any, samples: list[DetectionSample], device: Any
) -> tuple[list[Prediction], list[float]]:
    import torch

    dataset = _TorchDataset(samples, train=False)
    predictions: list[Prediction] = []
    latencies: list[float] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            image, _ = dataset[index]
            started = time.perf_counter()
            output = model([image.to(device)])[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000)
            predictions.append(
                Prediction(
                    boxes=[tuple(float(v) for v in row) for row in output["boxes"].cpu().tolist()],
                    scores=[float(v) for v in output["scores"].cpu().tolist()],
                )
            )
    return predictions, latencies


class _TorchDataset:
    """YOLO-layout samples as torchvision detection tensors (boxes xyxy, label 1 = acne)."""

    def __init__(self, samples: list[DetectionSample], train: bool) -> None:
        self._samples = samples
        self._train = train

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import torch
        from PIL import Image
        from torchvision.transforms.functional import hflip, to_tensor

        sample = self._samples[index]
        with Image.open(sample.image) as source:
            image = to_tensor(source.convert("RGB"))
        boxes = torch.tensor(sample.boxes, dtype=torch.float32).reshape(-1, 4)
        if self._train and random.random() < 0.5:
            image = hflip(image)
            width = float(sample.width)
            boxes = torch.stack(
                [width - boxes[:, 2], boxes[:, 1], width - boxes[:, 0], boxes[:, 3]], dim=1
            )
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return image, target


if __name__ == "__main__":
    main()
