"""The full record of a gate training run: machine-readable files and a Markdown report.

Every run leaves, without anyone having to remember it:

* ``runs/gate/<run_id>/`` — ``config.json`` (arguments, seed, environment, commit),
  ``dataset.json`` (the build report plus the manifest hash), a copy of the manifest,
  ``history.csv`` (per epoch), ``threshold-curve.csv``, ``metrics-val.json`` and, after
  ``evaluate_gate``, ``metrics-test.json``;
* ``acne-infra/docs/entrenamientos/filtro-acne-<run_id>.md`` — the same, in prose, VERSIONED (the
  ``runs/`` folder is not in Git), with the considerations and decisions behind the run.

The report is written by ``train_gate`` and completed by ``evaluate_gate``; the two never disagree
because both read the same JSON files.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

#: Where the versioned reports go: the infra repo's docs, next to the CRISP-DM phases.
DEFAULT_DOCS_DIR = Path("../acne-infra/docs/entrenamientos")

CONSIDERATIONS = """\
## Consideraciones y decisiones

- **Por qué un filtro y no reentrenar los modelos IGA.** Los tres clasificadores IGA (YOLOv8s-cls,
  EfficientNet-B2, ResNet-50) se entrenaron y evaluaron solo con ACNE04; su F1 es un resultado de
  la tesis. ACNE04 no contiene piel sana ni otras dermatosis, así que graduaban cualquier imagen
  como acné; `sin_lesiones` nunca se entrenó. El filtro va delante y deja intactos esos modelos y
  sus métricas.
- **Tres clases.** `acne` → se gradúa con el ensemble IGA; `sana` → la sugerencia es
  `sin_lesiones` con la probabilidad del filtro; `otra` → no se gradúa y el portal avisa de la
  incongruencia. **Las pecas (efélides) son una variante normal de la piel sana** y pertenecen a
  `sana`; Fitzpatrick17k no tiene esa clase y CFD aporta rostros sanos reales.
- **Decisión en dos pasos con prioridad al acné.** Si `p(acné) ≥ umbral`, es acné; si no, la más
  probable entre sana y otra. El umbral se fija en validación como el MÁS ALTO que mantiene la
  sensibilidad sobre acné ≥ objetivo, **con tope 0,5** (desde el 10/10/2026: la primera
  corrida eligió 0,998 con puntajes saturados y perdía acné de otras cámaras): un acné real
  detenido pierde su grado, que es peor que un falso acné que llega al dermatólogo de todos modos.
- **Riesgo de atajo (shortcut learning).** Cada fuente tiene su cámara: ACNE04 una clínica, CFD un
  estudio con fondo blanco y luz controlada, Fitzpatrick17k dos atlas dermatológicos. Un modelo
  puede aprender la fuente y no la piel. Mitigaciones: (1) recorte del rostro con el mismo YuNet
  del cribado antes de todo, que quita el fondo; (2) Fitzpatrick17k aporta acné Y otras condiciones
  de los mismos atlas; (3) recortes de media cara en el entrenamiento, porque CFD solo tiene fotos
  de frente y las tomas de perfil del protocolo no deben confundirse con acné por la pose; (4)
  métricas de test separadas por fuente: si la sensibilidad cae en el acné de Fitzpatrick17k, el
  filtro aprendió la cámara de ACNE04.
- **Partición por persona.** Una persona de CFD aparece hasta en cinco expresiones; la partición
  70/15/15 es por persona (grupo), nunca por archivo, para no evaluar con caras ya vistas.
- **Color intacto.** Ningún aumento toca el matiz: el eritema del acné frente a la mácula marrón o
  blanca es parte de la señal. Solo volteo, recorte, media cara, brillo y contraste ±15 %.
- **Licencias.** CFD: solo investigación no comercial; **sus fotos no se publican** (ninguna puede
  ir como figura de la tesis) ni se redistribuyen; se cita a Ma, Correll y Wittenbrink (2015),
  Ma, Kantner y Wittenbrink (2020) y Lakshmi et al. (2021). Fitzpatrick17k: Groh et al. (2021),
  imágenes de dermaamin.com y atlasdermatologico.com.br, uso académico con atribución. ACNE04:
  Wu et al. (2019).
- **Estado: candidato, no activo.** El checkpoint se guarda como `acne_gate.candidate.pt`, que el
  motor no carga. Solo se activa (renombrándolo a `acne_gate.pt`) después de comprobarlo con fotos
  reales de la clínica: las métricas de este reporte salen de las mismas fuentes con que se entrenó
  y no prueban que el filtro funcione con la cámara de la consulta.
- **Fitzpatrick17k incompleto (10/2026).** dermaamin.com, que aloja ~76 % de sus imágenes, responde
  404; solo atlasdermatologico.com.br sigue en línea, con apenas 24 imágenes de acné. El acné queda
  casi entero en ACNE04 y la piel sana entera en CFD, que es exactamente la situación en que un
  modelo aprende la cámara y no la piel. Una copia de Fitzpatrick17k con las imágenes (p. ej. en
  Kaggle) se integra con `--fitzpatrick-images` y recupera las 518 de acné. Estado de cada sitio en
  `dataset.json` → `fitzpatrick_hosts`.
- **Limitaciones conocidas.** CFD son fotos de estudio de frente, no clínicas; las tomas reales de
  perfil de piel sana solo se cubren con recortes de media cara. Fitzpatrick17k no tiene melasma
  ni pecas. Ninguna imagen proviene de la clínica del estudio: la validación con casos propios
  sigue pendiente y es la que confirma el comportamiento en consulta.
"""


def environment() -> dict[str, Any]:
    """Versions and hardware, so the run can be reproduced (RNF-03)."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import timm
        import torch

        info["torch"] = torch.__version__
        info["timm"] = timm.__version__
        info["cuda"] = torch.cuda.is_available()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    try:
        info["git_commit"] = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or None
        )
        info["git_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
    except OSError:
        info["git_commit"] = None
    return info


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def render(run_dir: Path) -> str:
    """The Markdown report of a run, from its JSON files."""
    config = read_json(run_dir / "config.json") or {}
    dataset = read_json(run_dir / "dataset.json") or {}
    val = read_json(run_dir / "metrics-val.json") or {}
    test = read_json(run_dir / "metrics-test.json")
    history = (run_dir / "history.csv").read_text(encoding="utf-8").strip().splitlines()

    args = config.get("arguments", {})
    env = config.get("environment", {})
    lines = [
        f"# Entrenamiento del filtro acné / piel sana / otra — `{config.get('version', '?')}`",
        "",
        f"- **Corrida:** `{config.get('run_id', '?')}` · inicio {config.get('started_at', '?')} · "
        f"fin {config.get('finished_at', '?')}",
        f"- **Checkpoint:** `{config.get('checkpoint', '?')}` (SHA-256 "
        f"`{config.get('checkpoint_sha256', '?')}`)",
        f"- **Registro completo:** `{_record_path(run_dir)}`",
        "",
        "## Datos",
        "",
        f"Manifiesto `{dataset.get('manifest', '?')}` "
        f"(SHA-256 `{dataset.get('manifest_sha256', '?')}`)"
        f", {dataset.get('total_images', '?')} imágenes. Partición por persona/grupo 70/15/15; "
        "grupos repartidos en más de un split: "
        f"**{dataset.get('groups_in_more_than_one_split', '?')}**.",
        "",
        "| split · fuente · etiqueta | n |",
        "|---|---|",
        *[
            f"| {k.replace(' | ', ' · ')} | {n} |"
            for k, n in (dataset.get("by_split_source_label") or {}).items()
        ],
        "",
        f"Rostros detectados por fuente (recorte YuNet): {dataset.get('face_found', {})}.",
        f"Descargas fallidas de Fitzpatrick17k: {len(dataset.get('failed_downloads') or [])}; "
        f"estado de los sitios: {dataset.get('fitzpatrick_hosts', {})}.",
        "",
        "Distribución por fototipo (Fitzpatrick17k) y por etnia (CFD) en `dataset.json` "
        "(`by_fitzpatrick`, `by_ethnicity`), y por condición en `by_condition`.",
        "",
        "## Modelo y entrenamiento",
        "",
        f"- Arquitectura: `{args.get('model')}` (timm), pesos ImageNet: "
        f"{'no' if args.get('no_pretrained') else 'sí'}; entrada {args.get('input_size')} px.",
        f"- Clases (orden de salida): {config.get('classes')}.",
        f"- Épocas {args.get('epochs')}, lote {args.get('batch_size')}, AdamW lr {args.get('lr')}, "
        "weight decay 1e-4, coseno; pérdida de entropía cruzada ponderada por frecuencia de clase.",
        f"- Pesos de clase: {config.get('class_weights')}.",
        "- Preprocesado: recorte cuadrado sobre el rostro (YuNet + 15 % de margen; recorte central "
        "si no hay rostro), normalización ImageNet. Aumentos: volteo horizontal, recorte 70-100 %, "
        "media cara (30 %), brillo y contraste ±15 %. Sin cambios de matiz. "
        "En entrenamiento las JPEG "
        "se decodifican reducidas (lado ≥ 1024 px, escalado DCT) por velocidad; la evaluación de "
        "test usa exactamente el preprocesado del servicio.",
        f"- Semilla {args.get('seed')}. Entorno: {env}.",
        f"- Época elegida: **{config.get('best_epoch')}** (mayor exactitud balanceada en "
        "validación).",
        "",
        "### Evolución por época",
        "",
        *_csv_table(history),
        "",
        "## Umbral de acné",
        "",
        "Objetivo de sensibilidad sobre acné: "
        f"**{args.get('target_sensitivity')}**. Umbral elegido: "
        f"**{val.get('threshold')}** (el más alto que cumple el objetivo en validación; curva "
        "completa en `threshold-curve.csv`).",
        "",
        "## Resultados en validación",
        "",
        *_metrics_block(val),
    ]
    if test:
        lines += ["", "## Resultados en test", "", *_metrics_block(test.get("overall", {}))]
        for title, key in (
            ("Por fuente", "by_source"),
            ("Por fototipo Fitzpatrick (Fitzpatrick17k)", "by_fitzpatrick"),
            ("Por etnia (CFD)", "by_ethnicity"),
            ("Por tercil de luminancia del rostro (CFD)", "by_luminance"),
        ):
            groups = test.get(key) or {}
            if not groups:
                continue
            lines += [
                "",
                f"### {title}",
                "",
                "| grupo | n | sens. acné | espec. acné | recall sana | recall otra | aviso |",
                "|---|---|---|---|---|---|---|",
            ]
            for name, m in groups.items():
                lines.append(
                    f"| {name} | {m.get('n')} | {_fmt(m.get('acne_sensitivity'))} | "
                    f"{_fmt(m.get('acne_specificity'))} | "
                    f"{_fmt(m.get('recall', {}).get('sana'))} | "
                    f"{_fmt(m.get('recall', {}).get('otra'))} | "
                    f"{_small_classes(m)} |"
                )
        rejection = test.get("by_condition") or {}
        if rejection:
            lines += [
                "",
                "### Qué hace el filtro con cada condición (test)",
                "",
                "| condición | n | → acné | → sana | → otra |",
                "|---|---|---|---|---|",
            ]
            for name, item in sorted(rejection.items(), key=lambda kv: -kv[1]["n"]):
                lines.append(
                    f"| {name} | {item['n']} | {_fmt(item['acne'])} | {_fmt(item['sana'])} | "
                    f"{_fmt(item['otra'])} |"
                )
    else:
        lines += [
            "",
            "## Resultados en test",
            "",
            "_Pendiente: `python -m training.gate.evaluate_gate`._",
        ]
    alternatives = sorted(run_dir.glob("metrics-test-umbral-*.json"))
    if alternatives:
        lines += [
            "",
            "## Análisis de sensibilidad del umbral (test)",
            "",
            "Mismo checkpoint y mismo test con otros umbrales de acné; no reemplaza el resultado "
            "principal.",
            "",
            "| umbral | sens. acné | espec. acné | sens. acné ACNE04 | sens. acné Fitzpatrick17k |",
            "|---|---|---|---|---|",
        ]
        for path in alternatives:
            alt = read_json(path) or {}
            overall = alt.get("overall", {})
            by_source = alt.get("by_source", {})
            lines.append(
                f"| {alt.get('threshold')} | {_fmt(overall.get('acne_sensitivity'))} | "
                f"{_fmt(overall.get('acne_specificity'))} | "
                f"{_fmt(by_source.get('acne04', {}).get('acne_sensitivity'))} | "
                f"{_fmt(by_source.get('fitzpatrick17k', {}).get('acne_sensitivity'))} |"
            )
    lines += ["", CONSIDERATIONS]
    return "\n".join(lines) + "\n"


def write_report(run_dir: Path, docs_dir: Path) -> Path | None:
    """Writes the Markdown next to the JSON files and into the versioned docs folder."""
    config = read_json(run_dir / "config.json") or {}
    text = render(run_dir)
    (run_dir / "REPORTE.md").write_text(text, encoding="utf-8")
    if not docs_dir.parent.is_dir():
        print(f"AVISO: no existe {docs_dir.parent}; el reporte queda solo en {run_dir}.")
        return None
    docs_dir.mkdir(parents=True, exist_ok=True)
    target = docs_dir / f"filtro-acne-{config.get('run_id', run_dir.name)}.md"
    target.write_text(text, encoding="utf-8")
    return target


def _small_classes(m: dict[str, Any]) -> str:
    """Warn about every class with fewer than 30 images in the group, not only the group size."""
    small = [f"{c} n={k}" for c, k in (m.get("counts") or {}).items() if 0 < k < 30]
    return f"solo descriptivo ({', '.join(small)})" if small else ""


def _record_path(run_dir: Path) -> str:
    return run_dir.as_posix() if run_dir.is_absolute() else f"acne-ai-engine/{run_dir.as_posix()}"


def _metrics_block(m: dict[str, Any]) -> list[str]:
    if not m:
        return ["_Sin datos._"]
    lines = [
        f"- n = {m.get('n')} ({m.get('counts')})",
        "- Sensibilidad sobre acné (acné que pasa a graduarse): "
        f"**{_fmt(m.get('acne_sensitivity'))}**",
        f"- Especificidad de acné (lo que no es acné y no se gradúa): "
        f"**{_fmt(m.get('acne_specificity'))}**",
        f"- AUC acné vs. resto: {_fmt(m.get('acne_auc'))}",
        f"- Exactitud balanceada (3 clases): {_fmt(m.get('balanced_accuracy'))}",
        f"- Recall por clase: {m.get('recall')}; precisión por clase: {m.get('precision')}",
    ]
    matrix = m.get("confusion_matrix")
    if matrix:
        classes = m.get("classes", ["otra", "sana", "acne"])
        lines += [
            "",
            "Matriz de confusión (filas = real, columnas = filtro):",
            "",
            "| real \\ filtro | " + " | ".join(classes) + " |",
            "|---|" + "---|" * len(classes),
            *[
                f"| {classes[i]} | " + " | ".join(str(v) for v in row) + " |"
                for i, row in enumerate(matrix)
            ],
        ]
    return lines


def _csv_table(lines: list[str]) -> list[str]:
    if not lines:
        return ["_Sin historial._"]
    header = lines[0].split(",")
    rows = [f"| {' | '.join(header)} |", "|" + "---|" * len(header)]
    rows += [f"| {' | '.join(line.split(','))} |" for line in lines[1:]]
    return rows


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return "—" if value != value else f"{value:.3f}"
    return str(value)
