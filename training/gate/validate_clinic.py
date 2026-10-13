"""Validates the ACTIVE acne gate on the clinic's own photographs, and records the result.

Covers: validación del filtro previo con fotos reales de consulta (pendiente de la §9 de
``acne-infra/docs/05_evaluation.md``).

    uv run python -m training.gate.validate_clinic --images archive/validacion-clinica

Put the photographs in one sub-folder per EXPECTED answer, as the dermatologist judges them:

    archive/validacion-clinica/
        acne/   — the patient has acne (any grade)
        sana/   — healthy skin, freckles included
        otra/   — another condition (vitiligo, melasma, rosacea…)

``archive/`` is gitignored: the photographs are patient data and never leave this machine (RNF-02,
Ley N.º 29733). Use anonymised file names — the per-image detail keeps them, but only under
``runs/`` (not versioned); the versioned report has counts only.

The gate is loaded exactly as the engine serves it (same checkpoint, same threshold resolution:
``--threshold`` > ``ACNE_GATE_THRESHOLD`` > the checkpoint's own), so what is measured is what the
clinic gets. The question that matters most is the first row: **how many real acne photographs
the gate lets through to grading.** An acne photograph stopped as ``sana`` or ``otra`` loses the
AI suggestion.

Writes ``runs/gate/clinic-validation/<run_id>/`` (``predictions.csv``, ``summary.json``) and
``acne-infra/docs/entrenamientos/validacion-clinica-<run_id>.md``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.models.acne_gate import GATE_CLASSES, TorchAcneGate
from training.gate import report
from training.gate.train_gate import gate_metrics

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
#: Sensitivity on acne below this and the gate should not stay active for the study.
MIN_ACCEPTABLE_SENSITIVITY = 0.95


def main() -> None:
    parser = argparse.ArgumentParser(description="Valida el filtro activo con fotos de la clínica.")
    parser.add_argument("--images", default="archive/validacion-clinica")
    parser.add_argument(
        "--checkpoint", help="Por defecto, el que sirve el motor (ACNE_GATE_CHECKPOINT)."
    )
    parser.add_argument("--threshold", type=float, help="Por defecto, el del motor.")
    parser.add_argument("--runs-dir", default="runs/gate/clinic-validation")
    parser.add_argument("--docs-dir", default=str(report.DEFAULT_DOCS_DIR))
    parser.add_argument("--label", default="", help="Nota corta para el reporte (p. ej. 'lote 1').")
    args = parser.parse_args()

    from app.config import get_settings

    settings = get_settings()
    checkpoint = args.checkpoint or settings.acne_gate_checkpoint
    threshold = args.threshold if args.threshold is not None else settings.acne_gate_threshold
    gate = TorchAcneGate(checkpoint, threshold)

    root = Path(args.images)
    samples = [
        (path, label)
        for label in GATE_CLASSES
        if (root / label).is_dir()
        for path in sorted((root / label).rglob("*"))
        if path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".")
    ]
    if not samples:
        raise SystemExit(
            f"No hay fotos en {root}/acne, {root}/sana ni {root}/otra. Ver la ayuda del script."
        )

    started = datetime.now(UTC)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    probabilities = []
    for path, expected in samples:
        result = gate.check(path.read_bytes(), "image/jpeg")
        probabilities.append([result.probabilities[c] for c in GATE_CLASSES])
        rows.append(
            {
                "file": path.relative_to(root).as_posix(),
                "expected": expected,
                "gate": result.label,
                "correct": result.label == expected,
                **{f"p_{c}": result.probabilities[c] for c in GATE_CLASSES},
            }
        )
    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = gate_metrics(
        np.array(probabilities),
        np.array([GATE_CLASSES.index(expected) for _, expected in samples]),
        gate.threshold,
    )
    verdict = _verdict(metrics)
    summary = {
        "run_id": run_id,
        "label": args.label,
        "date": started.isoformat(),
        "gate_version": gate.version,
        "weights_checksum": gate.weights_checksum,
        "threshold": gate.threshold,
        "images_folder": root.as_posix(),
        "metrics": metrics,
        "verdict": verdict,
        "environment": report.environment(),
    }
    report.write_json(run_dir / "summary.json", summary)
    text = _render(summary)
    (run_dir / "REPORTE.md").write_text(text, encoding="utf-8")
    docs = Path(args.docs_dir)
    if docs.parent.is_dir():
        docs.mkdir(parents=True, exist_ok=True)
        (docs / f"validacion-clinica-{run_id}.md").write_text(text, encoding="utf-8")

    counts = metrics["counts"]
    print(f"\n{gate.version} · umbral {gate.threshold} · {sum(counts.values())} fotos {counts}")
    print(f"Acné que pasa a graduarse (sensibilidad): {_fmt(metrics['acne_sensitivity'])}")
    print(f"Lo que no es acné y no se gradúa:          {_fmt(metrics['acne_specificity'])}")
    print(f"Acierto por clase: {metrics['recall']}")
    print(f"\n{verdict}\n\nDetalle por foto: {run_dir / 'predictions.csv'}")


def _verdict(metrics: dict[str, Any]) -> str:
    sensitivity = metrics["acne_sensitivity"]
    n_acne = metrics["counts"]["acne"]
    if n_acne == 0:
        return "SIN DATOS: no hay fotos con acné; no se puede juzgar lo más importante."
    note = " (pocas fotos de acné: resultado orientativo)" if n_acne < 20 else ""
    if sensitivity >= MIN_ACCEPTABLE_SENSITIVITY:
        return f"APTO{note}: el filtro deja pasar al menos el 95 % del acné real a la graduación."
    return (
        f"NO APTO{note}: el filtro detiene más del 5 % del acné real. Recomendación: "
        "ACNE_GATE_ENABLED=false hasta reentrenarlo con fotos de este tipo."
    )


def _render(summary: dict[str, Any]) -> str:
    m = summary["metrics"]
    classes = m["classes"]
    matrix = m["confusion_matrix"]
    return "\n".join(
        [
            f"# Validación clínica del filtro acné / piel sana / otra — `{summary['run_id']}`",
            "",
            f"- Filtro: `{summary['gate_version']}` (checksum `{summary['weights_checksum']}`), "
            f"umbral de acné {summary['threshold']}.",
            f"- Fecha: {summary['date']}. {summary['label']}",
            f"- Fotos: {m['counts']} (carpeta local, no versionada; datos de pacientes).",
            "",
            f"## Veredicto: {summary['verdict']}",
            "",
            f"- **Acné real que pasa a graduarse:** {_fmt(m['acne_sensitivity'])}",
            f"- Lo que no es acné y no se gradúa: {_fmt(m['acne_specificity'])}",
            f"- Acierto por clase: {m['recall']}",
            "",
            "Matriz de confusión (filas = lo que juzgó el especialista, columnas = el filtro):",
            "",
            "| esperado \\ filtro | " + " | ".join(classes) + " |",
            "|---|" + "---|" * len(classes),
            *[
                f"| {classes[i]} | " + " | ".join(str(v) for v in row) + " |"
                for i, row in enumerate(matrix)
            ],
            "",
            "## Consideraciones",
            "",
            "- Es la única medición del filtro con la cámara y la población reales de la consulta; "
            "las del entrenamiento salen de las mismas fuentes con que se entrenó.",
            "- Criterio: con sensibilidad sobre acné < 0,95 el filtro no debe quedar activo en el "
            "estudio (un acné detenido pierde la sugerencia de la IA).",
            "- El detalle por foto queda solo en `acne-ai-engine/runs/` (no versionado).",
            "",
        ]
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "—" if value != value else f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
