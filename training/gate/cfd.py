"""Chicago Face Database (CFD 3.0.1): images and norming data, for the gate and the fairness audit.

CFD — Ma, Correll & Wittenbrink (2015), Behavior Research Methods 47, 1122-1135; CFD-MR — Ma,
Kantner & Wittenbrink (2020); CFD-INDIA — Lakshmi et al. (2021). Healthy adult faces, frontal,
studio-lit, with self-reported ethnicity and age and the measured median luminance of the face.

TERMS OF USE, which bind every script that touches it: non-commercial research only; the photographs
and their data are NOT published (no CFD photograph may appear as a figure of the thesis) and NOT
redistributed — not to a repository, not to a file host, not to a cloud AI service. The folder is
gitignored and dockerignored for that reason. Every published use must cite the papers above.

The norming workbook is read with ``zipfile`` + ``xml.etree`` (no ``openpyxl`` dependency): the
sheets ``* U.S. Norming Data`` hold one header row starting with ``Model`` and one row per model.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
#: U.S. norms for every set, so the three share the same raters and the same measures.
NORMING_SHEETS = ("CFD U.S. Norming Data", "CFD-MR U.S. Norming Data", "CFD-I U.S. Norming Data")
IMAGE_FOLDERS = ("CFD", "CFD-MR", "CFD-INDIA")
#: First letter of the model ID (the file-naming scheme of the Read Me).
ETHNICITY = {
    "A": "asiatica_americana",
    "B": "negra",
    "C": "china",
    "I": "india",
    "J": "japonesa",
    "L": "latina",
    "M": "multirracial",
    "W": "blanca",
}
#: ``CFD-AF-200-228-N.jpg`` → model ``AF-200``, expression ``N``.
FILE_PATTERN = re.compile(r"^CFD-([A-Z]{2}-\d{3})-\d{3}(?:-\d)?-([A-Z]{1,2})\.jpg$", re.IGNORECASE)


@dataclass(frozen=True)
class CfdImage:
    path: Path
    model: str
    expression: str
    image_set: str
    ethnicity: str
    gender: str


@dataclass(frozen=True)
class CfdModel:
    model: str
    age: float | None
    luminance: float | None


def cfd_root(path: Path) -> Path:
    """Accepts the unzip folder or the ``CFD Version 3.0.1`` folder itself."""
    if (path / "Images").is_dir():
        return path
    for candidate in sorted(path.rglob("Images")):
        # The zip carries a `__MACOSX` copy of resource forks (`._*` files): never the real data.
        if "__MACOSX" in candidate.parts:
            continue
        if candidate.is_dir() and (candidate / "CFD").is_dir():
            return candidate.parent
    raise SystemExit(f"No se encontró la carpeta Images/CFD de la Chicago Face Database en {path}.")


def list_images(root: Path, expressions: set[str] | None = None) -> list[CfdImage]:
    """Every photograph of the three sets; ``expressions`` filters (``{"N"}`` = neutral only)."""
    images: list[CfdImage] = []
    for image_set in IMAGE_FOLDERS:
        folder = root / "Images" / image_set
        if not folder.is_dir():
            continue
        for path in sorted(folder.rglob("*.jpg")):
            match = FILE_PATTERN.match(path.name)
            if match is None or path.name.startswith("._"):
                continue
            model, expression = match.group(1).upper(), match.group(2).upper()
            if expressions is not None and expression not in expressions:
                continue
            images.append(
                CfdImage(
                    path=path,
                    model=model,
                    expression=expression,
                    image_set=image_set,
                    ethnicity=ETHNICITY.get(model[0], "desconocida"),
                    gender="F" if model[1] == "F" else "M",
                )
            )
    return images


def read_norming(root: Path) -> dict[str, CfdModel]:
    """Age and face luminance per model, from the U.S. norming sheets."""
    workbook = next(
        (p for p in root.glob("*Norming Data*.xlsx") if not p.name.startswith("._")), None
    )
    if workbook is None:
        return {}
    models: dict[str, CfdModel] = {}
    with zipfile.ZipFile(workbook) as archive:
        strings = _shared_strings(archive)
        for name, target in _sheets(archive).items():
            if name not in NORMING_SHEETS:
                continue
            rows = _rows(archive, target, strings)
            header_index = next(
                (i for i, row in enumerate(rows) if row.get("A", "").strip() == "Model"), None
            )
            if header_index is None:
                continue
            header = {column: value.strip() for column, value in rows[header_index].items()}
            by_name = {value: column for column, value in header.items()}
            for row in rows[header_index + 1 :]:
                model = _normalise_model(row.get(by_name.get("Model", "A"), ""))
                if model is None or model in models:
                    continue
                # Self-reported age is mostly blank; the raters' estimate fills it.
                age = _number(row.get(by_name.get("AgeSelf", ""), ""))
                if age is None:
                    age = _number(row.get(by_name.get("AgeRated", ""), ""))
                models[model] = CfdModel(
                    model=model,
                    age=age,
                    luminance=_number(row.get(by_name.get("LuminanceMedian", ""), "")),
                )
    return models


def _normalise_model(value: str) -> str | None:
    """``AF-200`` in the main sets; the India sheet writes ``IM600-101`` (model + image ID)."""
    match = re.search(r"([A-Z]{2})-?(\d{3})", value.upper())
    return f"{match.group(1)}-{match.group(2)}" if match else None


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(t.text or "" for t in item.iter(f"{{{NS['m']}}}t"))
        for item in root.findall("m:si", NS)
    ]


def _sheets(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {rel.get("Id"): rel.get("Target", "") for rel in rels}
    sheets = workbook.find("m:sheets", NS)
    result: dict[str, str] = {}
    for sheet in [] if sheets is None else sheets:
        target = targets.get(sheet.get(REL_ID), "")
        result[sheet.get("name", "")] = target.removeprefix("/xl/").removeprefix("xl/")
    return result


def _rows(archive: zipfile.ZipFile, target: str, strings: list[str]) -> list[dict[str, str]]:
    """Each row as {column letter: text}; empty cells are simply absent."""
    sheet = ET.fromstring(archive.read(f"xl/{target}"))
    data = sheet.find("m:sheetData", NS)
    rows: list[dict[str, str]] = []
    for row in [] if data is None else data.findall("m:row", NS):
        cells: dict[str, str] = {}
        for cell in row.findall("m:c", NS):
            column = re.sub(r"\d", "", cell.get("r", ""))
            value = cell.find("m:v", NS)
            if value is None or value.text is None:
                inline = cell.find("m:is", NS)
                if inline is not None:
                    cells[column] = "".join(t.text or "" for t in inline.iter(f"{{{NS['m']}}}t"))
                continue
            cells[column] = strings[int(value.text)] if cell.get("t") == "s" else value.text
        rows.append(cells)
    return rows
