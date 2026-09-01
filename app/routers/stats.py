"""Statistical contrasts. Covers: RF-OE1-04, RF-OE2-05, RF-OE3-05, RF-OE4-03."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas import PairedTestRequest, PairedTestResponse
from app.stats.tests import FUNCTION_VERSION, InsufficientDataError, run_paired_test

router = APIRouter(prefix="/stats", tags=["statistics"])


@router.post(
    "/paired-test",
    response_model=PairedTestResponse,
    summary="Shapiro-Wilk sobre las diferencias y luego t pareada o Wilcoxon.",
    description=(
        "Aplica la regla de decisión declarada en la tesis. Si Shapiro-Wilk no rechaza la "
        "normalidad de las diferencias pareadas se usa la t de Student para muestras "
        "relacionadas; en caso contrario, la prueba de rangos con signo de Wilcoxon. Para HE₃ el "
        "diseño es de una muestra contra una línea base declarada (RF-OE3-05)."
    ),
)
def paired_test(request: PairedTestRequest) -> PairedTestResponse:
    try:
        return run_paired_test(request)
    except InsufficientDataError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)
        ) from error


@router.get(
    "/version",
    summary="Versión de la función estadística, para la trazabilidad de RNF-05.",
)
def version() -> dict[str, str]:
    return {"function_version": FUNCTION_VERSION}
