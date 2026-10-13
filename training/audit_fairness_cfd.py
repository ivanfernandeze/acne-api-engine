"""Exploratory equity audit of the IGA models on healthy faces (Chicago Face Database).

Covers: auditoría de equidad por tono de piel (sección 8.2), que ACNE04 no permite porque no trae
edad ni fototipo.

    uv run python -m training.audit_fairness_cfd --cfd-dir cfd

Every CFD photograph is a HEALTHY face, so the right answer is ``sin_lesiones``. That class was
never trained (ACNE04 has none): YOLO cannot emit it and the timm members only do so from an
untrained output, so it is not a validated answer either. ``casi_sin_lesiones`` (1-5 lesions) is a
borderline miss; anything from ``leve`` up is a clear false positive. The audit runs the SERVED
ensemble (``get_classifier``, same preprocessing as ``/infer/single``, Grad-CAM off) over the
neutral photographs and asks one question: **does the grade rise with ethnicity or skin tone?**

Reported per self-reported ethnicity and per face-luminance tercile (CFD's measured median
luminance; low = darker skin): the class distribution, the mean ordinal grade, the share graded
``leve`` or worse; a Kruskal-Wallis test across groups and the Spearman correlation between
luminance and grade (``scipy``). It is EXPLORATORY: studio photographs are not clinical ones, and
the groups are those of a U.S. face database, not of the study's patients.

Writes ``runs/v1.0/fairness-cfd/<run_id>/`` (``predictions.csv``, local only: it names CFD models;
``summary.json``) and the aggregate report ``acne-infra/docs/entrenamientos/
auditoria-equidad-cfd-<run_id>.md`` (no image, no individual data — CFD terms of use).
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.gate import report

IGA_ORDER = ("sin_lesiones", "casi_sin_lesiones", "leve", "moderado", "severo")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoría de equidad de los modelos IGA con CFD.")
    parser.add_argument("--cfd-dir", default="cfd")
    parser.add_argument("--runs-dir", default="runs/v1.0/fairness-cfd")
    parser.add_argument("--docs-dir", default=str(report.DEFAULT_DOCS_DIR))
    parser.add_argument("--limit", type=int, default=0, help="Solo las primeras N (pruebas).")
    args = parser.parse_args()

    # The served models, without Grad-CAM (not needed, and it triples the time) and without the
    # acne gate: what is audited is what the IGA ensemble does on its own.
    os.environ["ENABLE_GRADCAM"] = "false"
    os.environ["ACNE_GATE_ENABLED"] = "false"
    from app.config import get_settings
    from app.models.registry import get_classifier
    from training.gate.cfd import cfd_root, list_images, read_norming

    get_settings.cache_clear()
    classifier = get_classifier()
    if not classifier.operational:
        raise SystemExit(
            "El motor no está sirviendo los modelos entrenados (MODEL_BACKEND=torch en .env): "
            "auditar el stub no diría nada."
        )

    started = datetime.now(UTC)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    root = cfd_root(Path(args.cfd_dir))
    norming = read_norming(root)
    images = list_images(root, expressions={"N"})
    if args.limit:
        images = images[: args.limit]

    rows: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        result = classifier.predict(image.path.read_bytes(), "image/jpeg")
        model = norming.get(image.model)
        rows.append(
            {
                "model": image.model,
                "ethnicity": image.ethnicity,
                "gender": image.gender,
                "luminance": None if model is None else model.luminance,
                "age": None if model is None else model.age,
                "iga_class": result.iga_class,
                "grade": IGA_ORDER.index(result.iga_class),
                "confidence": result.confidence,
            }
        )
        if index % 100 == 0:
            print(f"  {index}/{len(images)} rostros")

    with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    tercile = _terciles([r["luminance"] for r in rows if r["luminance"] is not None])
    for row in rows:
        row["luminance_band"] = None if row["luminance"] is None else tercile(row["luminance"])

    summary = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "model_version": classifier.version,
        "weights_checksum": classifier.weights_checksum,
        "n_images": len(rows),
        "expression": "N (neutral)",
        "overall": _group_stats(rows),
        "by_ethnicity": _by(rows, "ethnicity"),
        "by_luminance": _by(rows, "luminance_band"),
        "by_gender": _by(rows, "gender"),
        "tests": _tests(rows),
        "environment": report.environment(),
    }
    report.write_json(run_dir / "summary.json", summary)
    text = _render(summary)
    (run_dir / "REPORTE.md").write_text(text, encoding="utf-8")
    docs = Path(args.docs_dir)
    if docs.parent.is_dir():
        docs.mkdir(parents=True, exist_ok=True)
        (docs / f"auditoria-equidad-cfd-{run_id}.md").write_text(text, encoding="utf-8")
        print(f"Reporte versionado: {docs / f'auditoria-equidad-cfd-{run_id}.md'}")

    overall = summary["overall"]
    print(
        f"\n{len(rows)} rostros sanos · grado medio {overall['mean_grade']} · "
        f"graduados leve o peor: {overall['share_leve_or_worse']:.1%}"
    )
    for key in ("by_ethnicity", "by_luminance"):
        for name, stats in summary[key].items():
            print(
                f"  {name:<22} n={stats['n']:<4} grado medio {stats['mean_grade']}  "
                f"leve+ {stats['share_leve_or_worse']:.1%}"
            )
    print(f"Pruebas: {summary['tests']}\nRegistro: {run_dir}")


def _terciles(values: list[float]) -> Any:
    ordered = sorted(values)
    low, high = ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]

    def band(value: float) -> str:
        if value < low:
            return f"1_baja (<{low:.0f})"
        return f"2_media ({low:.0f}-{high:.0f})" if value < high else f"3_alta (>={high:.0f})"

    return band


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grades = [r["grade"] for r in rows]
    counts = Counter(r["iga_class"] for r in rows)
    return {
        "n": len(rows),
        "distribution": {c: counts.get(c, 0) for c in IGA_ORDER},
        "mean_grade": round(sum(grades) / len(grades), 3) if grades else None,
        "share_leve_or_worse": round(sum(g >= 2 for g in grades) / len(grades), 4) if grades else 0,
        "mean_confidence": round(sum(r["confidence"] for r in rows) / len(rows), 4)
        if rows
        else None,
    }


def _by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(key):
            groups[row[key]].append(row)
    return {name: _group_stats(items) for name, items in sorted(groups.items())}


def _tests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy import stats

    results: dict[str, Any] = {}
    for key in ("ethnicity", "luminance_band"):
        groups: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            if row.get(key):
                groups[row[key]].append(row["grade"])
        samples = [g for g in groups.values() if len(g) >= 5]
        if len(samples) >= 2 and len({x for s in samples for x in s}) > 1:
            h, p = stats.kruskal(*samples)
            results[f"kruskal_{key}"] = {"H": round(float(h), 4), "p": round(float(p), 6)}
        else:
            results[f"kruskal_{key}"] = "no aplicable (un solo grado observado o grupos pequeños)"
    paired = [(r["luminance"], r["grade"]) for r in rows if r["luminance"] is not None]
    if len(paired) > 2 and len({g for _, g in paired}) > 1:
        rho, p = stats.spearmanr([a for a, _ in paired], [b for _, b in paired])
        results["spearman_luminance_grade"] = {"rho": round(float(rho), 4), "p": round(float(p), 6)}
    return results


def _render(summary: dict[str, Any]) -> str:
    def table(groups: dict[str, dict[str, Any]]) -> list[str]:
        lines = [
            "| grupo | n | " + " | ".join(IGA_ORDER) + " | grado medio | leve o peor | aviso |",
            "|---|---|" + "---|" * len(IGA_ORDER) + "---|---|---|",
        ]
        for name, s in groups.items():
            dist = " | ".join(str(s["distribution"][c]) for c in IGA_ORDER)
            note = "n < 30: solo descriptivo" if s["n"] < 30 else ""
            lines.append(
                f"| {name} | {s['n']} | {dist} | {s['mean_grade']} | "
                f"{s['share_leve_or_worse']:.1%} | {note} |"
            )
        return lines

    overall = summary["overall"]
    return "\n".join(
        [
            f"# Auditoría exploratoria de equidad de los modelos IGA — `{summary['run_id']}`",
            "",
            f"- Modelo auditado: `{summary['model_version']}` (checksum "
            f"`{summary['weights_checksum']}`), Grad-CAM y filtro de acné desactivados.",
            f"- Datos: {summary['n_images']} fotografías neutrales de la Chicago Face Database "
            "(rostros sanos de adultos; CFD, CFD-MR y CFD-INDIA). Ninguna imagen se reproduce aquí "
            "(términos de uso de CFD).",
            f"- Fechas: {summary['started_at']} → {summary['finished_at']}. Entorno: "
            f"{summary['environment']}.",
            "",
            "## Qué se mide",
            "",
            "Todas son caras sanas: la respuesta correcta es `sin_lesiones`. Esa clase nunca se "
            "entrenó (ACNE04 no la tiene): YOLOv8s-cls no puede emitirla y los modelos timm solo "
            "la emiten desde una salida sin entrenar, así que tampoco es una respuesta validada. "
            "`casi_sin_lesiones` (1-5 lesiones) es un error leve y **todo lo que sea `leve` o más "
            "es un falso positivo claro**. La pregunta es si ese error **crece con la etnia o con "
            "el tono de piel** (luminancia mediana del rostro medida por CFD; tercil bajo = piel "
            "más oscura).",
            "",
            "## Global",
            "",
            f"- Grado medio {overall['mean_grade']} (0 = sin lesiones … 4 = severo); graduados "
            f"leve o peor: **{overall['share_leve_or_worse']:.1%}**; confianza media "
            f"{overall['mean_confidence']}.",
            f"- Distribución: {overall['distribution']}.",
            "",
            "## Por etnia autoinformada",
            "",
            *table(summary["by_ethnicity"]),
            "",
            "## Por tercil de luminancia del rostro",
            "",
            *table(summary["by_luminance"]),
            "",
            "## Por género",
            "",
            *table(summary["by_gender"]),
            "",
            "## Pruebas estadísticas",
            "",
            f"{summary['tests']}",
            "",
            "Kruskal-Wallis compara la distribución del grado entre grupos; Spearman, la relación "
            "entre luminancia y grado (rho < 0 significa que las pieles más oscuras reciben grados "
            "más altos). Con p < 0,05 la diferencia no es atribuible al azar.",
            "",
            "## Consideraciones",
            "",
            "- **Exploratoria.** Son fotos de estudio, de frente, con luz controlada; "
            "los pacientes "
            "del estudio se fotografían en consulta y de perfil. Los grupos son de una base "
            "estadounidense, no de la población de Trujillo.",
            "- **Mide solo falsos positivos en piel sana**, no la exactitud sobre acné real por "
            "fototipo, que exigiría fotos de acné etiquetadas con fototipo (Fitzpatrick17k aporta "
            "518, sin grado IGA).",
            "- **Mitigación en el sistema:** el filtro previo de piel sana responde "
            "estas caras como "
            "`sin_lesiones` antes de que lleguen a los modelos IGA; su equidad por etnia y "
            "luminancia se reporta en el registro de su entrenamiento.",
            "- **Licencia:** CFD solo para investigación no comercial; "
            "sin publicar ni redistribuir "
            "sus fotos. Citar Ma, Correll y Wittenbrink (2015); Ma, Kantner y Wittenbrink (2020); "
            "Lakshmi et al. (2021).",
            "",
        ]
    )


if __name__ == "__main__":
    main()
