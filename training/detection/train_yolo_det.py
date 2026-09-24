"""Train the YOLOv8 lesion detector on ``data/detection`` and score it on the frozen test split.

Covers: Fase 4-5 de CRISP-DM for the detector.

    uv run python -m training.detection.ingest_acne04_detection
    uv run python -m training.detection.train_yolo_det --config training/config.yaml

Single class (``acne``): ACNE04 marks every lesion with a box and no type, so what the model
learns is *where* and *how many*, which is the Hayashi count. Hue and saturation jitter stay off
(seccion 8.3) as they do for the classifiers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from training.detection.common import Prediction, evaluate_predictions, read_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena YOLOv8 (detección) sobre ACNE04.")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--data", default="./data/detection")
    parser.add_argument("--weights", default="yolov8s.pt", help="Pesos COCO de Ultralytics")
    parser.add_argument("--tag", default="yolov8s-det")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de conteo (motor)")
    parser.add_argument("--device", default=None, help="0 (GPU) | cpu; por defecto automático")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Falta ultralytics: uv pip install ultralytics") from error

    data = Path(args.data)
    data_yaml = data / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(
            f"No existe {data_yaml}. Ejecute antes: "
            "python -m training.detection.ingest_acne04_detection"
        )

    seed = int(config["seed"])
    tracking = Path(config["tracking_dir"]) / config["dataset_version"] / "detection"
    tracking.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.tag.replace('-', '_')}_lesiones.pt"
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if not args.skip_train:
        model = YOLO(args.weights)
        started = time.perf_counter()
        model.train(
            data=str(data_yaml),
            imgsz=args.imgsz,
            epochs=args.epochs,
            patience=args.patience,
            batch=args.batch,
            seed=seed,
            deterministic=True,
            workers=0,
            device=args.device,
            save_dir=str(tracking / f"{args.tag}-{run_id}"),
            project=str(tracking),
            name=f"{args.tag}-{run_id}",
            exist_ok=True,
            plots=True,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.10,
            fliplr=0.5,
            degrees=10.0,
            mosaic=1.0,
            close_mosaic=10,
            cos_lr=True,
        )
        run_dir = Path(model.trainer.save_dir)
        shutil.copy2(run_dir / "weights" / "best.pt", checkpoint_path)
        print(
            f"\nEntrenamiento: {time.perf_counter() - started:.0f} s · checkpoint {checkpoint_path}"
        )
    else:
        run_dir = Path(args.run_dir) if args.run_dir else None
        if not checkpoint_path.is_file():
            raise SystemExit(f"No hay checkpoint en {checkpoint_path}.")

    # --- Frozen test split, shared evaluator -------------------------------------------------
    detector = YOLO(str(checkpoint_path))
    samples = read_split(data, "test")
    predictions: list[Prediction] = []
    latencies: list[float] = []
    for sample in samples:
        started = time.perf_counter()
        result = detector.predict(
            str(sample.image), imgsz=args.imgsz, conf=0.001, device=args.device, verbose=False
        )[0]
        latencies.append((time.perf_counter() - started) * 1000)
        boxes = result.boxes
        predictions.append(
            Prediction(
                boxes=[tuple(float(v) for v in row) for row in boxes.xyxy.tolist()],
                scores=[float(v) for v in boxes.conf.tolist()],
            )
        )

    payload = evaluate_predictions(samples, predictions, conf_threshold=args.conf)
    payload.update(
        {
            "model": "yolov8-det",
            "tag": args.tag,
            "weights": args.weights,
            "imgsz": args.imgsz,
            "parameters": int(sum(p.numel() for p in detector.model.parameters())),
            "dataset_version": config["dataset_version"],
            "split": "test",
            "model_version": f"{args.tag}_{run_id}",
            "seed": seed,
            "inference_ms_median": round(sorted(latencies[1:])[len(latencies[1:]) // 2], 2)
            if len(latencies) > 2
            else None,
            "run_dir": str(run_dir) if run_dir is not None else None,
        }
    )
    output = tracking / f"metrics-detection-{args.tag}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print(payload, output)


def _print(payload: dict[str, Any], output: Path) -> None:
    print(f"\nDetector '{payload['tag']}' · test · n = {payload['n']}")
    print(f"  mAP50            : {payload['map50']:.4f}")
    print(f"  mAP50-95         : {payload['map50_95']:.4f}")
    print(f"  Precisión@{payload['conf_threshold']}   : {payload['precision_at_threshold']:.4f}")
    print(f"  Recall@{payload['conf_threshold']}      : {payload['recall_at_threshold']:.4f}")
    print(f"  MAE conteo       : {payload['count_mae']:.2f} lesiones")
    r = payload["count_pearson_r"]
    print(f"  r conteo         : {'—' if r is None else f'{r:.3f}'}")
    print(
        f"  Grado por conteo : exactitud {payload['count_grade_accuracy']:.3f} · "
        f"±1 {payload['count_grade_within_one']:.3f}"
    )
    print(f"Métricas escritas en {output}")


if __name__ == "__main__":
    main()
