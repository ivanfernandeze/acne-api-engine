"""Evaluates the gate on the test split and completes the run's record and report.

Covers: Fase 5 de CRISP-DM para el filtro previo.

    uv run python -m training.gate.evaluate_gate

Loads the checkpoint exactly as the engine serves it (``TorchAcneGate``, which crops the face with
YuNet itself) and reads every image as bytes, so the numbers are those of the deployed filter.
Reported, overall and per subgroup:

* sensitivity on acne (acne that goes on to be graded) and specificity (what is not acne and is
  not graded), the AUC of acne against the rest, recall per class and the confusion matrix;
* **per source** — a gate that only works on ACNE04 has learned the camera, not the skin;
* **per Fitzpatrick phototype** (Fitzpatrick17k: I-II, III-IV, V-VI), **per ethnicity** and **per
  face-luminance tercile** (CFD): the equity audit of the filter;
* what the filter does with each condition (share sent to acne / healthy / other).

Writes ``runs/gate/<run_id>/metrics-test.json`` and re-renders the run's report (``REPORTE.md`` and
``acne-infra/docs/entrenamientos/filtro-acne-<run_id>.md``).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from app.models.acne_gate import GATE_CLASSES, TorchAcneGate
from training.gate import report
from training.gate.train_gate import gate_metrics, load_manifest

PHOTOTYPE_BANDS = {"1": "I-II", "2": "I-II", "3": "III-IV", "4": "III-IV", "5": "V-VI", "6": "V-VI"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el filtro acné / piel sana / otra.")
    parser.add_argument("--manifest", default="data/gate/manifest.csv")
    parser.add_argument("--checkpoint", default="checkpoints/acne_gate.candidate.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--runs-dir", default="runs/gate")
    parser.add_argument("--docs-dir", default=str(report.DEFAULT_DOCS_DIR))
    parser.add_argument(
        "--threshold",
        type=float,
        help=(
            "Umbral de acné alternativo (análisis de sensibilidad). Se guarda aparte, en "
            "metrics-test-umbral-<t>.json, sin reemplazar el resultado del checkpoint."
        ),
    )
    args = parser.parse_args()

    import torch

    gate = TorchAcneGate(args.checkpoint, args.threshold)
    run_id = str(torch.load(args.checkpoint, map_location="cpu").get("run_id", "sin_run_id"))
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = [r for r in load_manifest(Path(args.manifest)) if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"No hay filas en el split {args.split}.")

    probabilities = []
    for index, row in enumerate(rows, start=1):
        result = gate.check(Path(row["path"]).read_bytes(), "image/jpeg")
        probabilities.append([result.probabilities[c] for c in GATE_CLASSES])
        if index % 250 == 0:
            print(f"  evaluadas {index}/{len(rows)}")
    probs = np.array(probabilities)
    targets = np.array([GATE_CLASSES.index(r["label"]) for r in rows])

    luminance_band = _luminance_terciles(rows)
    results: dict[str, Any] = {
        "version": gate.version,
        "weights_checksum": gate.weights_checksum,
        "split": args.split,
        "threshold": gate.threshold,
        "overall": gate_metrics(probs, targets, gate.threshold),
        "by_source": _grouped(rows, probs, targets, gate.threshold, lambda r: r["source"]),
        "by_fitzpatrick": _grouped(
            rows, probs, targets, gate.threshold, lambda r: PHOTOTYPE_BANDS.get(r["fitzpatrick"])
        ),
        "by_ethnicity": _grouped(
            rows, probs, targets, gate.threshold, lambda r: r.get("ethnicity") or None
        ),
        "by_luminance": _grouped(
            rows, probs, targets, gate.threshold, lambda r: luminance_band.get(r["path"])
        ),
        "by_condition": _by_condition(rows, probs, gate.threshold),
    }
    name = (
        "metrics-test.json"
        if args.threshold is None
        else f"metrics-test-umbral-{args.threshold:g}.json"
    )
    report.write_json(run_dir / name, results)
    docs = report.write_report(run_dir, Path(args.docs_dir))

    overall = results["overall"]
    print(
        f"\n{gate.version} · umbral {gate.threshold:.3f} · split {args.split} (n={overall['n']})\n"
        f"Sensibilidad sobre acné: {overall['acne_sensitivity']:.3f}\n"
        f"Especificidad de acné:   {overall['acne_specificity']:.3f}\n"
        f"Recall por clase:        {overall['recall']}\n"
    )
    for key in ("by_source", "by_fitzpatrick", "by_ethnicity", "by_luminance"):
        for name, m in results[key].items():
            print(
                f"  {key[3:]:<12} {name:<22} n={m['n']:<5} sens.acné {m['acne_sensitivity']}  "
                f"sana {m['recall']['sana']}  otra {m['recall']['otra']}"
            )
    print(f"\nRegistro: {run_dir / 'metrics-test.json'}")
    if docs is not None:
        print(f"Reporte versionado: {docs}")


def _grouped(
    rows: list[dict[str, str]],
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    key: Any,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        name = key(row)
        if name:
            groups[name].append(index)
    return {
        name: gate_metrics(probs[idx], targets[idx], threshold)
        for name, idx in sorted(groups.items())
    }


def _luminance_terciles(rows: list[dict[str, str]]) -> dict[str, str]:
    """CFD face luminance in three equal-count bands: low = darker skin, high = lighter."""
    values = sorted(float(r["luminance"]) for r in rows if r.get("luminance"))
    if len(values) < 3:
        return {}
    low, high = values[len(values) // 3], values[2 * len(values) // 3]
    bands = {}
    for row in rows:
        if not row.get("luminance"):
            continue
        value = float(row["luminance"])
        bands[row["path"]] = (
            f"1_baja (<{low:.0f})"
            if value < low
            else f"2_media ({low:.0f}-{high:.0f})"
            if value < high
            else f"3_alta (>={high:.0f})"
        )
    return bands


def _by_condition(
    rows: list[dict[str, str]], probs: np.ndarray, threshold: float
) -> dict[str, dict[str, Any]]:
    from training.gate.train_gate import decide_all

    predicted = decide_all(probs, threshold)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[f"{row['label']}: {row['original_label']}"].append(index)
    result = {}
    for name, idx in groups.items():
        chosen = predicted[idx]
        result[name] = {
            "n": len(idx),
            **{c: round(float(np.mean(chosen == i)), 4) for i, c in enumerate(GATE_CLASSES)},
        }
    return result


if __name__ == "__main__":
    main()
