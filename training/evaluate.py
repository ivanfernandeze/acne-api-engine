"""Evaluate a trained classifier on the frozen test split.

Covers: Fase 5 de CRISP-DM (seccion 8.5). Writes ``runs/<dataset_version>/metrics-<split>.json``,
which is exactly what ``POST /model/eval`` serves and what feeds ``REPORTE-EVALUACION-MODELO.pdf``,
plus ``metrics-<split>-<tag>.json`` so several models can be compared (``training.compare``).

    uv run python -m training.evaluate --config training/config.yaml
    uv run python -m training.evaluate --checkpoint checkpoints/resnet50_iga.pt --tag resnet50

``compute_metrics`` is the single definition of the metric set; ``train_yolo_cls`` and
``cross_validate`` reuse it so every model is scored by the same code.
"""

from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument(
        "--checkpoint", default=None, help="Por defecto <output_dir>/efficientnet_b2_iga.pt"
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Etiqueta del modelo para metrics-<split>-<tag>.json. "
            "Por defecto, model_name del checkpoint."
        ),
    )
    parser.add_argument(
        "--no-primary",
        action="store_true",
        help="No sobrescribir metrics-<split>.json (el archivo que sirve POST /model/eval).",
    )
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    try:
        import timm
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit("Falta el extra [torch]: uv pip install -e '.[torch]'") from error

    checkpoint_path = Path(args.checkpoint or Path(config["output_dir"]) / "efficientnet_b2_iga.pt")
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

    model_name = str(checkpoint.get("model_name", config["model"]["name"]))
    input_size = int(checkpoint.get("input_size", config["model"]["input_size"]))
    tag = args.tag or model_name
    model = timm.create_model(model_name, pretrained=False, num_classes=len(IGA_CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)

    loader = DataLoader(
        build_torch_dataset(subset, input_size, train=False),
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

    # Single-image latency, the way the engine serves it (batch of one). The first call pays
    # for kernel selection and is discarded.
    single = DataLoader(build_torch_dataset(subset[:21], input_size, train=False), batch_size=1)
    latencies: list[float] = []
    with torch.no_grad():
        for images, _ in single:
            images = images.to(device)
            started = time.perf_counter()
            model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000)
    latencies = sorted(latencies[1:])

    payload = compute_metrics(targets, predictions, probabilities)
    payload.update(
        {
            "inference_ms_median": round(latencies[len(latencies) // 2], 2) if latencies else None,
            "model_name": model_name,
            "tag": tag,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "input_size": input_size,
            "dataset_version": config["dataset_version"],
            "split": args.split,
            "model_version": checkpoint.get("version", "desconocida"),
            "seed": checkpoint.get("seed", config["seed"]),
            "run_id": checkpoint.get("version"),
            "n": len(targets),
        }
    )

    output_dir = Path(config["tracking_dir"]) / config["dataset_version"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"metrics-{args.split}-{tag}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.no_primary:
        (output_dir / f"metrics-{args.split}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    accuracy = payload["accuracy"]
    f1_macro = payload["f1_macro"]
    auc_roc_macro = payload["auc_roc_macro"]
    per_class = payload["per_class_metrics"]

    print(f"\nModelo '{model_name}' · split '{args.split}' · n = {len(targets)}")
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
    print(f"\nCriterio de éxito de la Fase 1 (F1 macro >= {TARGET_F1_MACRO:.3f}): {verdict}")
    print(f"Métricas escritas en {output_path}")
    print("Ya están disponibles en POST /model/eval y en REPORTE-EVALUACION-MODELO.pdf.")


def compute_metrics(
    targets: list[int], predictions: list[int], probabilities: list[list[float]]
) -> dict[str, Any]:
    """The section 8.5 metric set for one split, identical for every classifier compared.

    ``probabilities`` has one column per IGA class in ``IGA_CLASSES`` order. AUC-ROC is computed
    only over the classes present in ``targets``: a class with zero support (e.g. `sin_lesiones`
    while training on ACNE04 alone) has no positives, so its one-vs-rest curve is undefined and
    would drop the whole metric. ``within_one_grade`` is the share of images graded exactly or
    on a neighbouring grade — what a triage tool can live with — and ``far_errors`` the count that
    jump two grades or more.
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    labels = list(range(len(IGA_CLASSES)))
    matrix = confusion_matrix(targets, predictions, labels=labels)
    y_true = np.array(targets)
    y_pred = np.array(predictions)
    accuracy = float(np.mean(y_true == y_pred)) if len(targets) else 0.0
    f1_macro = float(f1_score(targets, predictions, average="macro", zero_division=0))

    auc_roc_macro: float | None = None
    present = sorted(set(targets))
    if len(present) > 1:
        try:
            scores = np.array(probabilities)[:, present]
            scores = scores / np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)
            auc_roc_macro = float(
                roc_auc_score(targets, scores, multi_class="ovr", average="macro", labels=present)
            )
        except ValueError:
            auc_roc_macro = None

    errors = int(np.sum(y_true != y_pred))
    far_errors = int(np.sum(np.abs(y_true - y_pred) >= 2))

    return {
        "accuracy": round(accuracy, 6),
        "f1_macro": round(f1_macro, 6),
        "auc_roc_macro": None if auc_roc_macro is None else round(auc_roc_macro, 6),
        "within_one_grade": round(1.0 - far_errors / len(targets), 6) if targets else 0.0,
        "far_errors": far_errors,
        "errors": errors,
        "confusion_matrix": matrix.tolist(),
        "per_class_metrics": _per_class_metrics(matrix),
    }


def _per_class_metrics(matrix: Any) -> list[dict[str, Any]]:
    total = int(matrix.sum())
    results: list[dict[str, Any]] = []

    for index, iga_class in enumerate(IGA_CLASSES):
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum()) - true_positive
        false_positive = int(matrix[:, index].sum()) - true_positive
        true_negative = total - true_positive - false_negative - false_positive

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else 0.0
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
