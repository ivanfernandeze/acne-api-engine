"""Model selection table: every ``metrics-test-<tag>.json`` side by side, one winner.

Covers: Fase 4-5 de CRISP-DM, seleccion de modelo.

    uv run python -m training.compare --config training/config.yaml

Reads ``runs/<dataset_version>/metrics-test-*.json`` (written by ``training.evaluate`` and
``training.train_yolo_cls``) and, when present, ``cv-*.json`` and each run's ``history.json`` for
the epoch time. Writes ``comparison.json`` and prints the Markdown table that ``04_modeling.md``
reproduces.

Selection rule, stated once here and nowhere else: highest **F1 macro on the frozen test split**;
a difference under ``TIE_MARGIN`` is a tie and the tie goes to EfficientNet-B2, the architecture
section 8.1 fixes. The rule is deliberately blind to validation scores — the split the recipe was
tuned on cannot also pick the winner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from app.domain.iga import IGA_CLASSES

TIE_MARGIN = 0.01
REFERENCE_TAG = "efficientnet_b2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara los clasificadores evaluados.")
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tracking = Path(config["tracking_dir"]) / config["dataset_version"]

    rows: list[dict[str, Any]] = []
    for path in sorted(tracking.glob("metrics-test-*.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        tag = str(metrics.get("tag") or path.stem.removeprefix("metrics-test-"))
        rows.append(
            {
                "tag": tag,
                "model_name": metrics.get("model_name", tag),
                "parameters": metrics.get("parameters"),
                "input_size": metrics.get("input_size"),
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "auc_roc_macro": metrics.get("auc_roc_macro"),
                "within_one_grade": metrics.get("within_one_grade"),
                "far_errors": metrics.get("far_errors"),
                "f1_per_class": {
                    entry["iga_class"]: entry["f1"] for entry in metrics["per_class_metrics"]
                },
                "inference_ms_median": metrics.get("inference_ms_median"),
                "seconds_per_epoch": _seconds_per_epoch(tracking, metrics),
                "cv": _cross_validation(tracking, tag),
                "model_version": metrics.get("model_version"),
                "n": metrics.get("n"),
            }
        )

    if not rows:
        raise SystemExit(f"No hay metrics-test-*.json en {tracking}.")

    rows.sort(key=lambda r: r["f1_macro"], reverse=True)
    winner = _select(rows)

    payload = {
        "dataset_version": config["dataset_version"],
        "rule": (
            f"Mayor F1 macro en el test congelado; diferencia < {TIE_MARGIN} es empate y gana "
            f"{REFERENCE_TAG} (Sección 8.1)."
        ),
        "winner": winner,
        "models": rows,
    }
    (tracking / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(_markdown(rows, winner))
    print(f"\nEscrito en {tracking / 'comparison.json'}")


def _select(rows: list[dict[str, Any]]) -> str:
    best = rows[0]
    reference = next((r for r in rows if r["tag"] == REFERENCE_TAG), None)
    if reference is not None and best["f1_macro"] - reference["f1_macro"] < TIE_MARGIN:
        return REFERENCE_TAG
    return str(best["tag"])


def _seconds_per_epoch(tracking: Path, metrics: dict[str, Any]) -> float | None:
    """Mean epoch time from the run's history.json (timm) or results.csv (Ultralytics)."""
    version = str(metrics.get("model_version") or "")
    run_id = version.rsplit("_", 1)[-1] if "_" in version else ""
    history = tracking / run_id / "history.json"
    if history.is_file():
        epochs = json.loads(history.read_text(encoding="utf-8"))
        return round(sum(e["seconds"] for e in epochs) / max(len(epochs), 1), 1)
    run_dir = metrics.get("run_dir")
    results = Path(run_dir) / "results.csv" if run_dir else None
    if results is not None and results.is_file():
        lines = [line for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
        header = [h.strip() for h in lines[0].split(",")]
        if "time" in header and len(lines) > 1:
            index = header.index("time")
            total = float(lines[-1].split(",")[index])
            return round(total / (len(lines) - 1), 1)
    return None


def _cross_validation(tracking: Path, tag: str) -> dict[str, float] | None:
    candidates = sorted(tracking.glob("cv-*.json"))
    for path in reversed(candidates):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_tag = payload.get("tag") or payload.get("config", {}).get("model", {}).get("name")
        if model_tag == tag:
            return {
                "f1_macro_mean": payload["f1_macro_mean"],
                "f1_macro_std": payload["f1_macro_std"],
            }
    return None


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}".replace(".", ",")
    return str(value)


def _markdown(rows: list[dict[str, Any]], winner: str) -> str:
    present = [c for c in IGA_CLASSES if any(r["f1_per_class"].get(c, 0) > 0 for r in rows)]
    header = (
        "| Modelo | Parámetros | Exactitud | F1 macro | AUC-ROC | ±1 grado | Errores >1 | "
        + " | ".join(f"F1 {c}" for c in present)
        + " | s/época | ms/inf. | CV F1 (5-fold) |"
    )
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for r in rows:
        mark = " **(servido)**" if r["tag"] == winner else ""
        params = f"{r['parameters'] / 1e6:.1f} M".replace(".", ",") if r["parameters"] else "—"
        cv = r["cv"]
        cv_text = f"{_fmt(cv['f1_macro_mean'])} ± {_fmt(cv['f1_macro_std'])}" if cv else "—"
        per_class = " | ".join(_fmt(r["f1_per_class"].get(c)) for c in present)
        lines.append(
            f"| {r['tag']}{mark} | {params} | {_fmt(r['accuracy'])} | "
            f"**{_fmt(r['f1_macro'])}** | {_fmt(r['auc_roc_macro'])} | "
            f"{_fmt(r['within_one_grade'])} | {_fmt(r['far_errors'])} | {per_class} | "
            f"{_fmt(r['seconds_per_epoch'], 1)} | {_fmt(r['inference_ms_median'], 0)} | {cv_text} |"
        )
    lines.append("")
    lines.append(
        f"Ganador según la regla (F1 macro en test, empate → {REFERENCE_TAG}): **{winner}**"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
