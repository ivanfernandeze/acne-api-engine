"""Stratified k-fold cross-validation with the same recipe as ``training.train``.

Covers: Fase 5 de CRISP-DM (seccion 8.5, ``evaluation.cross_validation_folds``).

    uv run python -m training.cross_validate --config training/config.yaml

The frozen test split is excluded: folds are drawn from train + val only, so the number reported
here never touches the images that back ``metrics-test.json``. With ~150 validation images a
single split moves several F1 points on luck alone; the mean and standard deviation over folds is
the figure the thesis can defend.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.domain.iga import IGA_CLASSES
from training.dataset import Sample, build_torch_dataset, load_manifest, stratified_split
from training.train import fit


def main() -> None:
    parser = argparse.ArgumentParser(description="Validación cruzada estratificada k-fold.")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--folds", type=int, default=None, help="Por defecto, config.evaluation")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    try:
        import timm
        import torch
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit("Falta el extra [torch]: uv pip install -e '.[torch]'") from error

    seed = int(config["seed"])
    folds = int(args.folds or config["evaluation"]["cross_validation_folds"])

    manifest = Path(args.manifest or Path(config["data_dir"]) / "manifest.csv")
    samples = load_manifest(manifest)
    splits = stratified_split(
        samples,
        (config["splits"]["train"], config["splits"]["val"], config["splits"]["test"]),
        seed,
    )
    pool: list[Sample] = sorted(splits["train"] + splits["val"], key=lambda s: str(s.path))
    labels = np.array([IGA_CLASSES.index(sample.iga_class) for sample in pool])
    print(f"Muestras en CV: {len(pool)} (test congelado excluido: {len(splits['test'])})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}  ·  folds: {folds}")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    fold_results: list[dict[str, Any]] = []

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_index, held_index) in enumerate(splitter.split(pool, labels), start=1):
        train_samples = [pool[i] for i in train_index]
        held_samples = [pool[i] for i in held_index]
        print(f"\n=== Fold {fold}/{folds}: train={len(train_samples)} val={len(held_samples)} ===")

        # Each fold gets its own seed so the sampler and CutMix draws differ, yet stay reproducible.
        result = fit(
            config, train_samples, held_samples, device, seed + fold, log_prefix=f"[f{fold}] "
        )

        model = timm.create_model(
            config["model"]["name"], pretrained=False, num_classes=len(IGA_CLASSES)
        )
        model.load_state_dict(result.best_state)
        model.eval().to(device)

        loader = DataLoader(
            build_torch_dataset(held_samples, int(config["model"]["input_size"]), train=False),
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=0,
        )
        targets: list[int] = []
        predictions: list[int] = []
        with torch.no_grad():
            for images, batch_labels in loader:
                logits = model(images.to(device))
                predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
                targets.extend(batch_labels.tolist())

        present = sorted(set(targets))
        per_class = f1_score(targets, predictions, labels=present, average=None, zero_division=0)
        fold_results.append(
            {
                "fold": fold,
                "n_train": len(train_samples),
                "n_val": len(held_samples),
                "best_epoch": result.best_epoch,
                "accuracy": round(float(np.mean(np.array(targets) == np.array(predictions))), 6),
                "f1_macro": round(
                    float(f1_score(targets, predictions, average="macro", zero_division=0)), 6
                ),
                "f1_per_class": {
                    IGA_CLASSES[index]: round(float(score), 6)
                    for index, score in zip(present, per_class, strict=True)
                },
            }
        )
        print(
            f"[f{fold}] acc={fold_results[-1]['accuracy']:.4f}  "
            f"f1_macro={fold_results[-1]['f1_macro']:.4f}  (época {result.best_epoch})"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    f1_values = np.array([fold["f1_macro"] for fold in fold_results])
    accuracy_values = np.array([fold["accuracy"] for fold in fold_results])
    payload = {
        "folds": folds,
        "seed": seed,
        "dataset_version": config["dataset_version"],
        "run_id": run_id,
        "n": len(pool),
        "f1_macro_mean": round(float(f1_values.mean()), 6),
        "f1_macro_std": round(float(f1_values.std(ddof=1)), 6) if folds > 1 else 0.0,
        "accuracy_mean": round(float(accuracy_values.mean()), 6),
        "accuracy_std": round(float(accuracy_values.std(ddof=1)), 6) if folds > 1 else 0.0,
        "per_fold": fold_results,
        "config": config,
    }

    output_dir = Path(config["tracking_dir"]) / config["dataset_version"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cv-{run_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nValidación cruzada {folds}-fold · n = {len(pool)}")
    print(f"  F1 macro : {payload['f1_macro_mean']:.4f} ± {payload['f1_macro_std']:.4f}")
    print(f"  Exactitud: {payload['accuracy_mean']:.4f} ± {payload['accuracy_std']:.4f}")
    print(f"Resultados escritos en {output_path}")


if __name__ == "__main__":
    main()
