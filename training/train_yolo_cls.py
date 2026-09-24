"""Train YOLOv8 in classification mode on the frozen split and score it like every other model.

Covers: Fase 4 de CRISP-DM, seleccion de modelo (EfficientNet-B2 vs ResNet-50 vs YOLOv8-cls).

    uv run python -m training.export_yolo_cls
    uv run python -m training.train_yolo_cls --config training/config.yaml

YOLOv8-cls is Ultralytics' image classifier (same backbone family as the detector, dense head).
It trains on ``data/yolo_cls/{train,val,test}/<class>/`` — the folders ``export_yolo_cls`` builds
from the same hash split — and is evaluated here with ``training.evaluate.compute_metrics`` on the
same 142 test images, so the comparison table in ``04_modeling.md`` is architecture-only.

Ultralytics' own augmentation defaults are overridden: hue and saturation jitter and RandAugment
are OFF (seccion 8.3 forbids altering skin tone), leaving flip, rotation, crop and brightness, the
same family used for the timm models. Ultralytics is AGPL-3.0; declared in ``04_modeling.md``.
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

from app.domain.iga import IGA_CLASSES
from training.evaluate import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena YOLOv8-cls sobre el split congelado.")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--data", default="./data/yolo_cls")
    parser.add_argument("--weights", default="yolov8s-cls.pt", help="Pesos ImageNet de Ultralytics")
    parser.add_argument("--tag", default="yolov8s-cls")
    parser.add_argument("--epochs", type=int, default=None, help="Por defecto training.epochs")
    parser.add_argument("--skip-train", action="store_true", help="Solo evaluar el checkpoint")
    parser.add_argument(
        "--run-dir", default=None, help="Con --skip-train: carpeta de la corrida (results.csv)"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=0,
        help="Validación cruzada k-fold sobre train+val (mismos folds que "
        "training.cross_validate); no toca el test congelado.",
    )
    args = parser.parse_args()
    if args.folds > 0:
        cross_validate(args)
        return

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Falta ultralytics: uv pip install ultralytics") from error

    data = Path(args.data)
    if not (data / "train").is_dir():
        raise SystemExit(
            f"No existe {data}/train. Ejecute antes: python -m training.export_yolo_cls"
        )

    seed = int(config["seed"])
    input_size = int(config["model"]["input_size"])
    epochs = int(args.epochs or config["training"]["epochs"])
    patience = int(config["training"]["early_stopping"]["patience"])
    aug = config.get("augmentation", {})
    crop_scale = aug.get("random_resized_crop_scale", [0.7, 1.0])

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tracking = Path(config["tracking_dir"]) / config["dataset_version"]
    checkpoint_dir = Path(config["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{args.tag.replace('-', '_')}_iga.pt"

    if not args.skip_train:
        model = YOLO(args.weights)
        started = time.perf_counter()
        model.train(
            data=str(data),
            imgsz=input_size,
            epochs=epochs,
            patience=patience,
            batch=int(config["training"]["batch_size"]),
            seed=seed,
            deterministic=True,
            workers=0,
            # Explicit: Ultralytics would otherwise nest it under <runs_dir>/classify/<project>.
            save_dir=str(tracking / f"{args.tag}-{run_id}"),
            project=str(tracking),
            name=f"{args.tag}-{run_id}",
            exist_ok=True,
            verbose=True,
            plots=True,
            # Section 8.3: never touch hue or saturation; brightness ±10 %, flip, rotation ±15°.
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=float(aug.get("brightness", 0.10)),
            fliplr=float(aug.get("horizontal_flip", 0.5)),
            flipud=0.0,
            degrees=float(aug.get("rotation_degrees", 15)),
            scale=float(1.0 - crop_scale[0]),
            erasing=0.0,
            auto_augment=None,
            dropout=float(config["model"].get("dropout", 0.3)),
            optimizer="AdamW",
            lr0=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            warmup_epochs=float(config["training"].get("warmup_epochs", 2)),
            cos_lr=True,
        )
        train_seconds = time.perf_counter() - started
        run_dir = Path(model.trainer.save_dir)
        best = run_dir / "weights" / "best.pt"
        shutil.copy2(best, checkpoint_path)
        print(f"\nEntrenamiento: {train_seconds:.0f} s · checkpoint copiado a {checkpoint_path}")
    else:
        run_dir = Path(args.run_dir) if args.run_dir else None
        if run_dir is not None and run_dir.name.startswith(f"{args.tag}-"):
            run_id = run_dir.name.removeprefix(f"{args.tag}-")
        if not checkpoint_path.is_file():
            raise SystemExit(f"No hay checkpoint en {checkpoint_path}.")

    # --- Evaluation on the frozen test split, with the shared metric code --------------------
    model = YOLO(str(checkpoint_path))
    names: dict[int, str] = model.names  # Ultralytics index -> folder name
    unknown = set(names.values()) - set(IGA_CLASSES)
    if unknown:
        raise SystemExit(f"Clases desconocidas en el modelo: {sorted(unknown)}")

    test_dir = data / "test"
    files = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    latencies: list[float] = []

    device = 0 if torch.cuda.is_available() else "cpu"
    for path in files:
        targets.append(IGA_CLASSES.index(path.parent.name))
        started = time.perf_counter()
        result = model.predict(str(path), imgsz=input_size, device=device, verbose=False)[0]
        latencies.append((time.perf_counter() - started) * 1000)
        probs_by_name = {names[i]: float(v) for i, v in enumerate(result.probs.data.tolist())}
        row = [probs_by_name.get(iga_class, 0.0) for iga_class in IGA_CLASSES]
        probabilities.append(row)
        predictions.append(max(range(len(row)), key=row.__getitem__))

    payload = compute_metrics(targets, predictions, probabilities)
    parameters = int(sum(p.numel() for p in model.model.parameters()))
    payload.update(
        {
            "model_name": args.tag,
            "tag": args.tag,
            "parameters": parameters,
            "input_size": input_size,
            "dataset_version": config["dataset_version"],
            "split": "test",
            "model_version": f"{args.tag}_{run_id}",
            "seed": seed,
            "run_id": f"{args.tag}_{run_id}",
            "n": len(targets),
            # Warm-up excluded: the first call pays for CUDA context and kernel selection.
            "inference_ms_median": round(sorted(latencies[1:])[len(latencies[1:]) // 2], 2)
            if len(latencies) > 2
            else None,
            "run_dir": str(run_dir) if run_dir is not None else None,
        }
    )
    tracking.mkdir(parents=True, exist_ok=True)
    output = tracking / f"metrics-test-{args.tag}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nModelo '{args.tag}' · split 'test' · n = {len(targets)}")
    print(f"  Exactitud      : {payload['accuracy']:.4f}")
    print(f"  F1 macro       : {payload['f1_macro']:.4f}")
    auc = payload["auc_roc_macro"]
    print(f"  AUC-ROC macro  : {'—' if auc is None else f'{auc:.4f}'}")
    for metrics in payload["per_class_metrics"]:
        print(
            f"  {metrics['iga_class']:<20} n={metrics['support']:<4} "
            f"prec={metrics['precision']:.3f} sens={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
        )
    print(f"Métricas escritas en {output}")
    print("Compare con:  uv run python -m training.compare")


def cross_validate(args: argparse.Namespace) -> None:
    """k-fold on train + val with the exact folds of ``training.cross_validate``.

    Each fold materialises its own ``train``/``val`` folders (hard links) under
    ``data/yolo_cls_cv/fold<k>`` and is trained from the ImageNet weights with the same recipe as
    the single run; the held-out fold is scored with ``compute_metrics``. Writes
    ``runs/<dataset_version>/cv-<run_id>.json`` carrying ``tag`` so ``training.compare`` can pair
    it with the test-split file.
    """
    import os
    import shutil

    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from ultralytics import YOLO

    from training.dataset import load_manifest, stratified_split

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = int(config["seed"])
    input_size = int(config["model"]["input_size"])
    epochs = int(args.epochs or config["training"]["epochs"])
    patience = int(config["training"]["early_stopping"]["patience"])
    aug = config.get("augmentation", {})
    crop_scale = aug.get("random_resized_crop_scale", [0.7, 1.0])
    tracking = Path(config["tracking_dir"]) / config["dataset_version"]
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    manifest = Path(config["data_dir"]) / "manifest.csv"
    samples = load_manifest(manifest)
    splits = stratified_split(
        samples,
        (config["splits"]["train"], config["splits"]["val"], config["splits"]["test"]),
        seed,
    )
    pool = sorted(splits["train"] + splits["val"], key=lambda s: str(s.path))
    labels = np.array([IGA_CLASSES.index(s.iga_class) for s in pool])
    print(f"Muestras en CV: {len(pool)} (test congelado excluido: {len(splits['test'])})")

    cv_root = Path("./data/yolo_cls_cv")
    if cv_root.exists():
        shutil.rmtree(cv_root)

    fold_results: list[dict[str, Any]] = []
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
    for fold, (train_index, held_index) in enumerate(splitter.split(pool, labels), start=1):
        fold_dir = cv_root / f"fold{fold}"
        for split_name, indices in (("train", train_index), ("val", held_index)):
            for i in indices:
                sample = pool[i]
                target = fold_dir / split_name / sample.iga_class / sample.path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(sample.path, target)
        print(f"\n=== Fold {fold}/{args.folds}: train={len(train_index)} val={len(held_index)} ===")

        model = YOLO(args.weights)
        model.train(
            data=str(fold_dir),
            imgsz=input_size,
            epochs=epochs,
            patience=patience,
            batch=int(config["training"]["batch_size"]),
            seed=seed + fold,
            deterministic=True,
            workers=0,
            save_dir=str(tracking / f"cv-{args.tag}-{run_id}" / f"fold{fold}"),
            project=str(tracking),
            name=f"cv-{args.tag}-{run_id}-fold{fold}",
            exist_ok=True,
            verbose=False,
            plots=False,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=float(aug.get("brightness", 0.10)),
            fliplr=float(aug.get("horizontal_flip", 0.5)),
            flipud=0.0,
            degrees=float(aug.get("rotation_degrees", 15)),
            scale=float(1.0 - crop_scale[0]),
            erasing=0.0,
            auto_augment=None,
            dropout=float(config["model"].get("dropout", 0.3)),
            optimizer="AdamW",
            lr0=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            warmup_epochs=float(config["training"].get("warmup_epochs", 2)),
            cos_lr=True,
        )
        best = YOLO(str(Path(model.trainer.save_dir) / "weights" / "best.pt"))
        names = dict(best.names)
        targets: list[int] = []
        predictions: list[int] = []
        probabilities: list[list[float]] = []
        for i in held_index:
            sample = pool[i]
            targets.append(IGA_CLASSES.index(sample.iga_class))
            result = best.predict(str(sample.path), imgsz=input_size, verbose=False)[0]
            by_name = {names[j]: float(v) for j, v in enumerate(result.probs.data.tolist())}
            row = [by_name.get(c, 0.0) for c in IGA_CLASSES]
            probabilities.append(row)
            predictions.append(max(range(len(row)), key=row.__getitem__))
        metrics = compute_metrics(targets, predictions, probabilities)
        fold_results.append(
            {
                "fold": fold,
                "n_train": len(train_index),
                "n_val": len(held_index),
                "best_epoch": int(getattr(model.trainer, "best_epoch", -1)),
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "f1_per_class": {
                    e["iga_class"]: e["f1"] for e in metrics["per_class_metrics"] if e["support"]
                },
            }
        )
        print(f"[f{fold}] acc={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}")

    f1 = np.array([r["f1_macro"] for r in fold_results])
    acc = np.array([r["accuracy"] for r in fold_results])
    payload = {
        "tag": args.tag,
        "folds": args.folds,
        "seed": seed,
        "dataset_version": config["dataset_version"],
        "run_id": run_id,
        "n": len(pool),
        "f1_macro_mean": round(float(f1.mean()), 6),
        "f1_macro_std": round(float(f1.std(ddof=1)), 6) if args.folds > 1 else 0.0,
        "accuracy_mean": round(float(acc.mean()), 6),
        "accuracy_std": round(float(acc.std(ddof=1)), 6) if args.folds > 1 else 0.0,
        "per_fold": fold_results,
        "config": {"model": {"name": args.tag}, "weights": args.weights, "input_size": input_size},
    }
    output = tracking / f"cv-{run_id}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nValidación cruzada {args.folds}-fold · n = {len(pool)}")
    print(f"  F1 macro : {payload['f1_macro_mean']:.4f} ± {payload['f1_macro_std']:.4f}")
    print(f"  Exactitud: {payload['accuracy_mean']:.4f} ± {payload['accuracy_std']:.4f}")
    print(f"Resultados escritos en {output}")


if __name__ == "__main__":
    main()
