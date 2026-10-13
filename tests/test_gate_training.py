"""The gate's data pipeline: no person in two splits, CFD read correctly, the threshold honest.

No CFD photograph and no row of its norming data is used: the workbook is a synthetic one with the
same layout (CFD terms of use forbid redistributing either).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from training.gate.build_dataset import _split_of
from training.gate.cfd import FILE_PATTERN, _normalise_model, read_norming
from training.gate.train_gate import gate_metrics, pick_threshold

SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def test_a_group_always_lands_in_the_same_split() -> None:
    # The five expressions of one CFD model share the group, so they share the split.
    splits = {_split_of("cfd:AF-200", seed=42) for _ in range(5)}
    assert len(splits) == 1


def test_the_split_is_roughly_70_15_15() -> None:
    counts = {"train": 0, "val": 0, "test": 0}
    for index in range(3000):
        counts[_split_of(f"src:{index}", seed=42)] += 1
    assert 0.65 < counts["train"] / 3000 < 0.75
    assert 0.11 < counts["val"] / 3000 < 0.19
    assert 0.11 < counts["test"] / 3000 < 0.19


@pytest.mark.parametrize(
    ("name", "model", "expression"),
    [
        ("CFD-AF-200-228-N.jpg", "AF-200", "N"),
        ("CFD-WM-001-014-HC.jpg", "WM-001", "HC"),
        ("CFD-IF-600-101-2-N.jpg", "IF-600", "N"),
    ],
)
def test_cfd_file_names_give_model_and_expression(name: str, model: str, expression: str) -> None:
    match = FILE_PATTERN.match(name)
    assert match is not None
    assert match.group(1) == model
    assert match.group(2) == expression


def test_cfd_model_ids_are_normalised_across_sheets() -> None:
    assert _normalise_model("AF-200") == "AF-200"
    # The India sheet writes model and image together, without the hyphen.
    assert _normalise_model("IM600-101") == "IM-600"
    assert _normalise_model("Model") is None


def _synthetic_workbook(path: Path) -> None:
    strings = ["Model", "AgeSelf", "AgeRated", "LuminanceMedian", "AF-200"]
    shared = "".join(f"<si><t>{s}</t></si>" for s in strings)
    sheet = (
        f'<worksheet xmlns="{SHEET_NS}"><sheetData>'
        '<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>'
        '<c r="C1" t="s"><v>2</v></c><c r="D1" t="s"><v>3</v></c></row>'
        # AgeSelf left blank on purpose: the rated age must fill it.
        '<row r="2"><c r="A2" t="s"><v>4</v></c><c r="C2"><v>27.5</v></c>'
        '<c r="D2"><v>140.25</v></c></row>'
        "</sheetData></worksheet>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{SHEET_NS}" xmlns:r="{REL_NS}"><sheets>'
            '<sheet name="CFD U.S. Norming Data" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="ws"/></Relationships>',
        )
        archive.writestr("xl/sharedStrings.xml", f'<sst xmlns="{SHEET_NS}">{shared}</sst>')
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


def test_the_norming_workbook_is_read_without_openpyxl(tmp_path: Path) -> None:
    _synthetic_workbook(tmp_path / "CFD 3.0.1 Norming Data and Codebook.xlsx")

    models = read_norming(tmp_path)

    assert set(models) == {"AF-200"}
    assert models["AF-200"].luminance == pytest.approx(140.25)
    assert models["AF-200"].age == pytest.approx(27.5)


def test_the_threshold_meets_the_sensitivity_target() -> None:
    rng = np.random.default_rng(0)
    # Columns in GATE_CLASSES order (otra, sana, acne); 40 acne images, 60 others.
    acne = rng.dirichlet([1, 1, 4], size=40)
    others = rng.dirichlet([3, 3, 1], size=60)
    probabilities = np.vstack([acne, others])
    targets = np.array([2] * 40 + [0] * 30 + [1] * 30)

    threshold, curve = pick_threshold(probabilities, targets, target_sensitivity=0.97)
    metrics = gate_metrics(probabilities, targets, threshold)

    assert metrics["acne_sensitivity"] >= 0.97
    assert len(curve) == 99
    # Truncated to 4 decimals, never rounded up past the lowest kept acne score.
    assert threshold == pytest.approx(round(threshold, 4))


def test_the_clinic_verdict_follows_the_acne_sensitivity() -> None:
    from training.gate.validate_clinic import _verdict

    def metrics(sensitivity: float, n_acne: int) -> dict[str, object]:
        return {"acne_sensitivity": sensitivity, "counts": {"otra": 5, "sana": 5, "acne": n_acne}}

    assert _verdict(metrics(0.97, 30)).startswith("APTO:")
    assert _verdict(metrics(0.80, 30)).startswith("NO APTO:")
    assert "orientativo" in _verdict(metrics(1.0, 5))
    assert _verdict(metrics(float("nan"), 0)).startswith("SIN DATOS")
