"""Builds the gate's dataset (acne / healthy skin / other condition), each image traced to source.

Covers: Fase 3 de CRISP-DM para el filtro previo (las clases IGA no cambian).

    uv run python -m training.gate.build_dataset \\
        --fitzpatrick-csv fitzapick17k/fitzpatrick17k.csv --download \\
        --cfd-dir cfd

Labels (``GATE_CLASSES`` of ``app/models/acne_gate.py``):

* ``acne`` — ACNE04 (``data/raw/<grado>/``, every grade) and the acne labels of Fitzpatrick17k
  (``acne``, ``acne vulgaris``: 518 images with phototype).
* ``sana`` — the Chicago Face Database: 826 healthy adults, 1 440 photographs (neutral and four
  expressions). Freckles, when present, are a normal variant of healthy skin and stay here.
* ``otra`` — every other condition of Fitzpatrick17k (vitiligo, psoriasis, rosacea, folliculitis,
  lupus…), at most ``--max-per-label`` images per condition so 113 conditions stay balanced and the
  download stays in the thousands, not 16 000.

Decisions taken here, and why (they are copied into the training report):

* **Split by person, not by file.** A CFD model appears in up to five expressions; scattering them
  over train and test would test the gate on faces it has already seen. ``group`` is the model ID
  for CFD, the image hash for Fitzpatrick17k and the file for ACNE04 (it has no subject IDs).
* **Face boxes are computed once here** with the screener's YuNet (``face_box``) and stored in the
  manifest, so training crops exactly what the service will crop. Without a face (Fitzpatrick17k
  close-ups of other body areas) the box is empty and the centred square is used.
* **Shortcut risk.** Every source has its own camera: ACNE04 one clinic, CFD one studio with a
  white wall, Fitzpatrick17k two atlases. A gate can learn the source instead of the skin. The face
  crop removes the background, Fitzpatrick17k contributes acne AND non-acne from the same atlases,
  and the evaluation reports every metric per source.

Writes ``data/gate/manifest.csv`` and ``data/gate/dataset-report.json`` (counts per split, source,
label, condition, phototype and ethnicity; failed downloads), which the training report embeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FITZPATRICK_COLUMNS = {"md5hash", "label", "url", "fitzpatrick_scale"}
SPLITS = (("train", 70), ("val", 15), ("test", 15))
#: Parallel downloads: enough to hide the atlas's ~1,5 s per image, few enough to be polite.
DOWNLOAD_WORKERS = 6
FIELDS = [
    "path",
    "label",
    "source",
    "original_label",
    "group",
    "fitzpatrick",
    "ethnicity",
    "luminance",
    "age",
    "face_box",
    "split",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Arma el dataset del filtro (acné/sana/otra).")
    parser.add_argument("--acne04-dir", default="data/raw", help="ACNE04 ingerido (todo es acné).")
    parser.add_argument("--fitzpatrick-csv", help="CSV de Fitzpatrick17k.")
    parser.add_argument("--fitzpatrick-images", default="archive/gate/fitzpatrick17k")
    parser.add_argument(
        "--download", action="store_true", help="Descargar las imágenes que falten."
    )
    parser.add_argument(
        "--acne-labels",
        default="acne,acne vulgaris",
        help="Etiquetas de Fitzpatrick17k que cuentan como acné, separadas por comas.",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=60,
        help="Máximo de imágenes por condición NO acné de Fitzpatrick17k (0 = todas).",
    )
    parser.add_argument("--cfd-dir", help="Chicago Face Database (piel sana).")
    parser.add_argument("--healthy-dir", action="append", default=[], metavar="NOMBRE=RUTA")
    parser.add_argument("--other-dir", action="append", default=[], metavar="NOMBRE=RUTA")
    parser.add_argument("--acne-dir", action="append", default=[], metavar="NOMBRE=RUTA")
    parser.add_argument(
        "--holdout-source",
        action="append",
        default=[],
        help="Fuente entera al split de test (para medir si el filtro aprendió la cámara).",
    )
    parser.add_argument(
        "--no-face-boxes", action="store_true", help="No calcular cajas de rostro (pruebas)."
    )
    parser.add_argument("--output", default="data/gate/manifest.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    failed_downloads: list[dict[str, str]] = []
    host_report: dict[str, object] = {}
    rows: list[dict[str, str]] = []
    rows += _folder_rows(Path(args.acne04_dir), "acne04", "acne")

    if args.fitzpatrick_csv:
        acne_labels = {label.strip().lower() for label in args.acne_labels.split(",") if label}
        rows += _fitzpatrick_rows(
            Path(args.fitzpatrick_csv),
            Path(args.fitzpatrick_images),
            acne_labels,
            max_per_label=args.max_per_label,
            download=args.download,
            seed=args.seed,
            failed=failed_downloads,
            host_report=host_report,
        )
    if args.cfd_dir:
        rows += _cfd_rows(Path(args.cfd_dir))

    for spec, label in (
        *((s, "sana") for s in args.healthy_dir),
        *((s, "otra") for s in args.other_dir),
        *((s, "acne") for s in args.acne_dir),
    ):
        name, path = _parse_spec(spec)
        rows += _folder_rows(Path(path), name, label)

    if not args.no_face_boxes:
        _add_face_boxes(rows)

    for row in rows:
        row["split"] = (
            "test"
            if row["source"] in args.holdout_source
            else _split_of(f"{row['source']}:{row['group']}", args.seed)
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report = _report(rows, args, failed_downloads, output)
    report["fitzpatrick_hosts"] = host_report
    report_path = output.with_name("dataset-report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_report(report)
    print(f"\nManifiesto: {output} ({len(rows)} imágenes)\nReporte: {report_path}")


# --- sources ---------------------------------------------------------------------------------


def _row(path: Path, label: str, source: str, original_label: str, group: str) -> dict[str, str]:
    return {
        "path": path.as_posix(),
        "label": label,
        "source": source,
        "original_label": original_label,
        "group": group,
        "fitzpatrick": "",
        "ethnicity": "",
        "luminance": "",
        "age": "",
        "face_box": "",
    }


def _parse_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise SystemExit(f"Formato NOMBRE=RUTA esperado, recibido: {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def _folder_rows(root: Path, source: str, label: str) -> list[dict[str, str]]:
    if not root.is_dir():
        raise SystemExit(f"No existe la carpeta {root} (fuente {source}).")
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or path.name.startswith("._"):
            continue
        # The sub-folder is the original class when there is one (ACNE04 grade, a condition).
        original = path.parent.name if path.parent != root else source
        rows.append(_row(path, label, source, original, group=path.stem))
    return rows


def _cfd_rows(root: Path) -> list[dict[str, str]]:
    from training.gate.cfd import cfd_root, list_images, read_norming

    base = cfd_root(root)
    norming = read_norming(base)
    rows = []
    for image in list_images(base):
        row = _row(image.path, "sana", "cfd", f"sana_{image.expression}", group=image.model)
        row["ethnicity"] = image.ethnicity
        model = norming.get(image.model)
        if model is not None:
            row["luminance"] = "" if model.luminance is None else f"{model.luminance:.2f}"
            row["age"] = "" if model.age is None else f"{model.age:.1f}"
        rows.append(row)
    return rows


def _fitzpatrick_rows(
    csv_path: Path,
    images_dir: Path,
    acne_labels: set[str],
    max_per_label: int,
    download: bool,
    seed: int,
    failed: list[dict[str, str]],
    host_report: dict[str, object],
) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = FITZPATRICK_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"El CSV de Fitzpatrick17k no tiene las columnas {sorted(missing)}.")
        records = list(reader)

    labels = Counter(record["label"].strip().lower() for record in records)
    unknown = acne_labels - set(labels)
    if unknown:
        print(f"AVISO: etiquetas de acné que no aparecen en el CSV: {sorted(unknown)}")
    print(
        "Fitzpatrick17k -> ACNÉ:",
        {label: labels[label] for label in sorted(acne_labels & set(labels))},
    )

    # Every acne image; up to `max_per_label` images of each other condition that CAN be obtained.
    # The candidates of each label are shuffled reproducibly and tried in order until the quota is
    # filled, so a dead host (dermaamin.com answers 404 for its 12 631 images since 2026) does not
    # leave a condition empty while its images on the other atlas are still online.
    by_label: dict[str, list[dict[str, str]]] = {}
    for record in records:
        by_label.setdefault(record["label"].strip().lower(), []).append(record)

    images_dir.mkdir(parents=True, exist_ok=True)
    # Copies of Fitzpatrick17k with the images included (e.g. on Kaggle) name each file after its
    # md5hash, in any sub-folder: index them so a mirror replaces the dead URLs.
    local = {
        path.stem.lower(): path
        for path in images_dir.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    }
    print(f"Fitzpatrick17k: {len(local)} imágenes ya disponibles en {images_dir}.")
    hosts = _HostHealth()
    obtained: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for position, label in enumerate(sorted(by_label), start=1):
            pending = sorted(by_label[label], key=lambda r: r["md5hash"])
            random.Random(f"{seed}:{label}").shuffle(pending)
            quota = None if label in acne_labels or max_per_label <= 0 else max_per_label
            taken = 0
            # In batches of exactly what is still missing, downloaded in parallel; a failed image
            # is replaced by the next candidate in the following batch.
            while pending and (quota is None or taken < quota):
                need = len(pending) if quota is None else quota - taken
                batch, pending = pending[:need], pending[need:]
                to_fetch: list[tuple[dict[str, str], Path, str]] = []
                for record in batch:
                    target = local.get(
                        record["md5hash"].lower(), images_dir / f"{record['md5hash']}.jpg"
                    )
                    if target.is_file():
                        record["_target"] = target.as_posix()
                        obtained.append(record)
                        taken += 1
                        continue
                    url = record["url"].strip()
                    if download and url and not hosts.is_dead(url):
                        to_fetch.append((record, target, url))
                results = pool.map(lambda item: _download(item[2], item[1]), to_fetch)
                for (record, target, url), ok in zip(to_fetch, results, strict=True):
                    hosts.record(url, ok)
                    if ok:
                        record["_target"] = target.as_posix()
                        obtained.append(record)
                        taken += 1
                    else:
                        failed.append(
                            {"md5hash": record["md5hash"], "label": record["label"], "url": url}
                        )
            if position % 10 == 0:
                print(
                    f"  condiciones revisadas {position}/{len(by_label)}; imágenes {len(obtained)}"
                )
    host_report.update(hosts.summary())
    print("Estado de los sitios de Fitzpatrick17k:", hosts.summary())

    rows = []
    for record in obtained:
        target = Path(record["_target"])
        label = record["label"].strip().lower()
        row = _row(
            target,
            "acne" if label in acne_labels else "otra",
            "fitzpatrick17k",
            label,
            group=record["md5hash"],
        )
        scale = str(record.get("fitzpatrick_scale", "")).strip()
        row["fitzpatrick"] = scale if scale in {"1", "2", "3", "4", "5", "6"} else ""
        rows.append(row)
    if failed:
        print(f"Fitzpatrick17k: {len(failed)} imágenes no se pudieron descargar; se omiten.")
    return rows


class _HostHealth:
    """Downloads, and gives up on a host after ``DEAD_AFTER`` consecutive failures."""

    DEAD_AFTER = 25

    def __init__(self) -> None:
        self.ok: Counter[str] = Counter()
        self.failed: Counter[str] = Counter()
        self.streak: Counter[str] = Counter()
        self.dead: set[str] = set()

    @staticmethod
    def host(url: str) -> str:
        return urllib.parse.urlparse(url).netloc

    def is_dead(self, url: str) -> bool:
        return self.host(url) in self.dead

    def record(self, url: str, ok: bool) -> None:
        host = self.host(url)
        if ok:
            self.ok[host] += 1
            self.streak[host] = 0
            return
        self.failed[host] += 1
        self.streak[host] += 1
        if host not in self.dead and self.streak[host] >= self.DEAD_AFTER and self.ok[host] == 0:
            print(f"AVISO: {host} no responde ({self.DEAD_AFTER} fallos seguidos); se omite.")
            self.dead.add(host)

    def summary(self) -> dict[str, dict[str, object]]:
        hosts = set(self.ok) | set(self.failed)
        return {
            host: {"ok": self.ok[host], "fallidas": self.failed[host], "caido": host in self.dead}
            for host in sorted(hosts)
        }


def _download(url: str, target: Path) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "acne-ai-engine/gate-dataset"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read()
    except Exception:
        return False
    if len(data) < 1024:
        return False
    target.write_bytes(data)
    time.sleep(0.05)  # be gentle with the hosting sites
    return True


def _add_face_boxes(rows: list[dict[str, str]]) -> None:
    """YuNet box per image, the same detector and rules the service uses (``face_box``)."""
    from PIL import Image

    from app.models.acne_gate import face_box, load_screener

    screener = load_screener()
    if screener is None:
        print("AVISO: sin OpenCV no hay recorte de rostro; se usará el recorte central.")
        return
    for index, row in enumerate(rows, start=1):
        try:
            with Image.open(row["path"]) as source:
                box = face_box(source.convert("RGB"), screener)
        except Exception:
            box = None
        row["face_box"] = "" if box is None else ",".join(str(v) for v in box)
        if index % 500 == 0:
            print(f"  rostros detectados en {index}/{len(rows)} imágenes")


def _split_of(key: str, seed: int) -> str:
    """Deterministic 70/15/15 from the GROUP: a person never lands in two splits."""
    bucket = int(hashlib.sha1(f"{seed}:{key}".encode()).hexdigest(), 16) % 100
    cumulative = 0
    for name, share in SPLITS:
        cumulative += share
        if bucket < cumulative:
            return name
    return "test"


# --- report ----------------------------------------------------------------------------------


def _report(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    failed: list[dict[str, str]],
    manifest: Path,
) -> dict[str, object]:
    def count(*keys: str) -> dict[str, int]:
        counter = Counter(" | ".join(row[k] or "—" for k in keys) for row in rows)
        return dict(sorted(counter.items()))

    groups_per_split: dict[str, set[str]] = {}
    for row in rows:
        groups_per_split.setdefault(f"{row['source']}:{row['group']}", set()).add(row["split"])
    leaks = sum(1 for splits in groups_per_split.values() if len(splits) > 1)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "manifest": manifest.as_posix(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "arguments": {k: v for k, v in vars(args).items()},
        "total_images": len(rows),
        "split_shares_percent": dict(SPLITS),
        "by_split_label": count("split", "label"),
        "by_split_source_label": count("split", "source", "label"),
        "by_condition": count("label", "original_label"),
        "by_fitzpatrick": count("label", "fitzpatrick"),
        "by_ethnicity": count("label", "ethnicity"),
        "face_found": {
            source: f"{sum(1 for r in rows if r['source'] == source and r['face_box'])}"
            f"/{sum(1 for r in rows if r['source'] == source)}"
            for source in sorted({r["source"] for r in rows})
        },
        "groups_in_more_than_one_split": leaks,
        "failed_downloads": failed,
    }


def _print_report(report: dict[str, object]) -> None:
    print("\nsplit | fuente | etiqueta -> n")
    for key, n in report["by_split_source_label"].items():  # type: ignore[union-attr]
        print(f"  {key}: {n}")
    print(f"\nPersonas/grupos en más de un split: {report['groups_in_more_than_one_split']}")
    print("Rostros detectados por fuente:", report["face_found"])


if __name__ == "__main__":
    main()
