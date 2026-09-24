"""Score the served ensemble on the frozen test split, with the same evaluator as its members.

Covers: Fase 5 de CRISP-DM for the model actually served (section 8.1, revision 17/09/2026).

    uv run python -m training.evaluate_ensemble --config training/config.yaml

Reads the members and weights from ``ENSEMBLE_MEMBERS`` (``app/config.py``, i.e. ``.env``), so
what is scored is exactly what ``/infer/single`` runs, and writes
``runs/<dataset_version>/metrics-test-ensemble.json`` plus, unless ``--no-primary``, the
``metrics-test.json`` that ``POST /model/eval`` serves. Also records how often the members agreed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from app.config import get_settings
from app.domain.iga import IGA_CLASSES
from app.models.ensemble_backend import EnsembleClassifier, parse_members
from training.dataset import load_manifest, stratified_split
from training.evaluate import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el ensemble servido sobre el test.")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--members", default=None, help="Por defecto ENSEMBLE_MEMBERS de .env")
    parser.add_argument("--no-primary", action="store_true")
    parser.add_argument(
        "--rule",
        default=None,
        choices=["mas_confiable", "voto_ponderado"],
        help="Por defecto ENSEMBLE_RULE de .env",
    )
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    settings = get_settings()
    specs = parse_members(args.members or settings.ensemble_members)

    samples = load_manifest(Path(config["data_dir"]) / "manifest.csv")
    splits = stratified_split(
        samples,
        (config["splits"]["train"], config["splits"]["val"], config["splits"]["test"]),
        int(config["seed"]),
    )
    subset = splits[args.split]

    # Grad-CAM off: it costs a backward pass per member and plays no part in the metrics.
    rule = args.rule or settings.ensemble_rule
    ensemble = EnsembleClassifier(
        specs,
        seed=int(config["seed"]),
        input_size=int(config["model"]["input_size"]),
        enable_gradcam=False,
        rule=rule,
    )
    tag = "ensemble-max" if rule == "mas_confiable" else "ensemble-avg"

    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    latencies: list[float] = []
    agreements: Counter[str] = Counter()
    member_hits: Counter[str] = Counter()

    for sample in subset:
        image_bytes = sample.path.read_bytes()
        started = time.perf_counter()
        result = ensemble.predict(image_bytes, "image/png")
        latencies.append((time.perf_counter() - started) * 1000)
        target = IGA_CLASSES.index(sample.iga_class)
        targets.append(target)
        predictions.append(IGA_CLASSES.index(result.iga_class))
        probabilities.append([result.class_probs[c] for c in IGA_CLASSES])
        agreements[result.agreement or "n/a"] += 1
        for member in result.members:
            member_hits[member.tag] += int(member.iga_class == sample.iga_class)

    payload = compute_metrics(targets, predictions, probabilities)
    latencies = sorted(latencies[1:])
    payload.update(
        {
            "model_name": "ensemble",
            "tag": tag,
            "rule": rule,
            "members": [
                {"tag": s.tag, "arch": s.arch, "weight": s.weight, "checkpoint": s.checkpoint_path}
                for s in specs
            ],
            "parameters": None,
            "input_size": int(config["model"]["input_size"]),
            "dataset_version": config["dataset_version"],
            "split": args.split,
            "model_version": ensemble.version,
            "weights_checksum": ensemble.weights_checksum,
            "seed": int(config["seed"]),
            "run_id": ensemble.version,
            "n": len(targets),
            "inference_ms_median": round(latencies[len(latencies) // 2], 2) if latencies else None,
            "agreement": {k: v for k, v in sorted(agreements.items())},
            "member_accuracy": {
                tag: round(hits / len(subset), 6) for tag, hits in member_hits.items()
            },
        }
    )

    output_dir = Path(config["tracking_dir"]) / config["dataset_version"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"metrics-{args.split}-{tag}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.no_primary:
        (output_dir / f"metrics-{args.split}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    print(f"\nEnsemble {ensemble.version} · split '{args.split}' · n = {len(targets)}")
    print(f"  Exactitud      : {payload['accuracy']:.4f}")
    print(f"  F1 macro       : {payload['f1_macro']:.4f}")
    auc = payload["auc_roc_macro"]
    print(f"  AUC-ROC macro  : {'—' if auc is None else f'{auc:.4f}'}")
    print(f"  ±1 grado       : {payload['within_one_grade']:.4f}")
    print(f"  Acuerdo        : {dict(agreements)}")
    print(f"  Exactitud por miembro: {payload['member_accuracy']}")
    for metrics in payload["per_class_metrics"]:
        print(
            f"  {metrics['iga_class']:<20} n={metrics['support']:<4} "
            f"prec={metrics['precision']:.3f} sens={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
        )
    print(f"Métricas escritas en {output}")


if __name__ == "__main__":
    main()
