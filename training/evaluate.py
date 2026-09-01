"""Evaluate the trained model on the frozen test split.

Covers: Fase 5 de CRISP-DM (seccion 8.5). Writes ``runs/<dataset_version>/metrics-test.json``,
which is exactly what ``POST /model/eval`` serves and what feeds ``REPORTE-EVALUACION-MODELO.pdf``.

    uv run python -m training.evaluate --config training/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from app.domain.iga import IGA_CLASSES
from training.dataset import build_torch_dataset, load_manifest, stratified_split

#: Fase 1 success criterion: 85 % of the F1 macro reported by Watanabe (2025).
TARGET_F1_MACRO = 0.885 * 0.85


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el modelo sobre el test congelado.")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    try:
        import numpy as np
        import timm
        import torch
        from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit("Falta el extra [torch]: uv pip install -e '.[torch]'") from error

    checkpoint_path = Path(config["output_dir"]) / "efficientnet_b2_iga.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"No hay checkpoint en {checkpoint_path}. Entrene primero:\n"
            "  uv run python -m training.train"
        )

    manifest = Path(args.manifest or Path(config["data_dir"]) / "manifest.csv")
    samples = load_manifest(manifest)
    splits = stratified_split(
        samples,
        (config["splits"]["train"], config["splits"]["val"], config["splits"]["test"]),
        int(config["seed"]),
    )
    subset = splits[args.split]
    if not subset:
        raise SystemExit(f"El split '{args.split}' está vacío.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = timm.create_model(
        config["model"]["name"], pretrained=False, num_classes=len(IGA_CLASSES)
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)

    loader = DataLoader(
        build_torch_dataset(subset, int(config["model"]["input_size"]), train=False),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            probabilities.extend(probs.cpu().tolist())
            predictions.extend(torch.argmax(probs, dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    matrix = confusion_matrix(targets, predictions, labels=list(range(len(IGA_CLASSES))))
    accuracy = float(np.mean(np.array(targets) == np.array(predictions)))
    f1_macro = float(f1_score(targets, predictions, average="macro", zero_division=0))

    # AUC-ROC needs at least two classes present in the split.
    auc_roc_macro: float | None = None
    if len(set(targets)) > 1:
        try:
            auc_roc_macro = float(
                roc_auc_score(
                    targets,
                    np.array(probabilities),
                    multi_class="ovr",
                    average="macro",
                    labels=list(range(len(IGA_CLASSES))),
                )
            )
        except ValueError:
            auc_roc_macro = None

    per_class = _per_class_metrics(matrix)

    payload = {
        "accuracy": round(accuracy, 6),
        "f1_macro": round(f1_macro, 6),
        "auc_roc_macro": None if auc_roc_macro is None else round(auc_roc_macro, 6),
        "confusion_matrix": matrix.tolist(),
        "per_class_metrics": per_class,
        "dataset_version": config["dataset_version"],
        "split": args.split,
        "model_version": checkpoint.get("version", "desconocida"),
        "seed": checkpoint.get("seed", config["seed"]),
        "run_id": checkpoint.get("version"),
        "n": len(targets),
    }

    output_dir = Path(config["tracking_dir"]) / config["dataset_version"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"metrics-{args.split}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nSplit '{args.split}' · n = {len(targets)}")
    print(f"  Exactitud      : {accuracy:.4f}")
    print(f"  F1 macro       : {f1_macro:.4f}")
    print(f"  AUC-ROC macro  : {'—' if auc_roc_macro is None else f'{auc_roc_macro:.4f}'}")
    print("\nMétricas por clase:")
    for metrics in per_class:
        print(
            f"  {metrics['iga_class']:<20} n={metrics['support']:<4} "
            f"prec={metrics['precision']:.3f} sens={metrics['recall']:.3f} "
            f"esp={metrics['specificity']:.3f} F1={metrics['f1']:.3f}"
        )

    verdict = "CUMPLE" if f1_macro >= TARGET_F1_MACRO else "NO CUMPLE"
    print(
        f"\nCriterio de éxito de la Fase 1 (F1 macro >= {TARGET_F1_MACRO:.3f}): {verdict}"
    )
    print(f"Métricas escritas en {output_path}")
    print("Ya están disponibles en POST /model/eval y en REPORTE-EVALUACION-MODELO.pdf.")


def _per_class_metrics(matrix: Any) -> list[dict[str, Any]]:
    total = int(matrix.sum())
    results: list[dict[str, Any]] = []

    for index, iga_class in enumerate(IGA_CLASSES):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum()) - true_positive
        false_positive = int(matrix[:, index].sum()) - true_positive
        true_negative = total - true_positive - false_negative - false_positive

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        specificity = (
            true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        results.append(
            {
                "iga_class": iga_class,
                "support": true_positive + false_negative,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "specificity": round(specificity, 6),
                "f1": round(f1, 6),
            }
        )
    return results


if __name__ == "__main__":
    main()
