"""IGA and GAGS scales.

Mirror of ``acne-api/src/common/domain/iga.ts``. The API is the source of truth for the domain;
this module exists because the engine must not depend on the orchestrator at runtime. Any change
to the class list must be made in both places and noted in the CHANGELOG of each repository.

Covers: OE2 (RF-OE2-01, RF-OE2-04).
"""

from __future__ import annotations

from typing import Final, Literal

IgaClass = Literal["sin_lesiones", "casi_sin_lesiones", "leve", "moderado", "severo"]

IGA_CLASSES: Final[tuple[IgaClass, ...]] = (
    "sin_lesiones",
    "casi_sin_lesiones",
    "leve",
    "moderado",
    "severo",
)

IGA_LABELS: Final[dict[IgaClass, str]] = {
    "sin_lesiones": "Sin lesiones",
    "casi_sin_lesiones": "Casi sin lesiones",
    "leve": "Leve",
    "moderado": "Moderado",
    "severo": "Severo",
}

#: Ordinal rank; IGA is an ordinal scale, so class distance is meaningful in error analysis.
IGA_ORDINAL: Final[dict[IgaClass, int]] = {
    "sin_lesiones": 0,
    "casi_sin_lesiones": 1,
    "leve": 2,
    "moderado": 3,
    "severo": 4,
}

#: Spanish codes since the 2026-09-05 restructuring. IGA was already Spanish, and carrying the two
#: clinical scales in different languages inside the same row was the kind of inconsistency an
#: examiner notices first. Changed in lockstep with ``acne-api/src/common/domain/iga.ts``.
GagsGrade = Literal["sin_acne", "leve", "moderado", "severo", "muy_severo"]

GAGS_GRADES: Final[tuple[GagsGrade, ...]] = (
    "sin_acne",
    "leve",
    "moderado",
    "severo",
    "muy_severo",
)

GAGS_LABELS: Final[dict[GagsGrade, str]] = {
    "sin_acne": "Sin acné",
    "leve": "Leve",
    "moderado": "Moderado",
    "severo": "Severo",
    "muy_severo": "Muy severo",
}

GAGS_MAX_SCORE: Final[int] = 44


def gags_grade_from_score(score: int) -> GagsGrade:
    """Map a GAGS score (0-44) to its band, per the standard cut-offs (Vasam, 2023)."""
    if score <= 0:
        return "sin_acne"
    if score <= 18:
        return "leve"
    if score <= 30:
        return "moderado"
    if score <= 38:
        return "severo"
    return "muy_severo"


#: Approximate GAGS score at the centre of each IGA band, used to derive the secondary scale
#: from an IGA prediction. It is a presentational convenience, not a second measurement.
IGA_TO_GAGS_CENTRE: Final[dict[IgaClass, int]] = {
    "sin_lesiones": 0,
    "casi_sin_lesiones": 8,
    "leve": 15,
    "moderado": 25,
    "severo": 36,
}
