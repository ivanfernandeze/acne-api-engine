"""IIA checklist. Covers: RF-TR-04, section 9.1."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from app.config import get_settings
from app.models.registry import get_classifier
from app.schemas import IiaCheckResponse
from app.services.telemetry import telemetry

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get(
    "/iia-check",
    response_model=IiaCheckResponse,
    summary="Ficha técnica de operatividad del modelo — valida IIA = 1.",
)
def iia_check() -> IiaCheckResponse:
    """Report whether the independent variable is actually implemented.

    This endpoint is the instrument that validates IIA = 1 in the thesis. It returns
    ``operational: false`` whenever the deterministic stub is serving, and lists the concrete
    blockers, so the coverage matrix and the reports downgrade themselves automatically instead of
    depending on someone remembering that the model is not trained yet.
    """
    settings = get_settings()
    classifier = get_classifier()
    snapshot = telemetry.snapshot()

    warnings: list[str] = []
    if not classifier.operational:
        warnings.append(
            "El motor opera con el clasificador stub determinista: las predicciones NO provienen "
            "de una CNN entrenada, por lo que la variable independiente no está implementada "
            "(IIA = 0)."
        )
        warnings.append(
            "Para habilitar IIA = 1: entrene EfficientNet-B2 (training/train.py), coloque el "
            "checkpoint en MODEL_CHECKPOINT_PATH y arranque con MODEL_BACKEND=torch."
        )
    if classifier.weights_checksum is None and classifier.operational:
        warnings.append("El modelo está activo pero no expone checksum de pesos (RNF-03).")

    p95 = snapshot["p95_ms"]
    if isinstance(p95, float) and p95 > 8000:
        warnings.append(
            f"La latencia p95 del motor es de {p95} ms y excede el máximo de 8000 ms que RNF-01 "
            "fija para CPU."
        )

    return IiaCheckResponse(
        operational=classifier.operational,
        model_backend=classifier.backend,  # type: ignore[arg-type]
        version=classifier.version,
        weights_checksum=classifier.weights_checksum,
        latency_median_24h_ms=snapshot["median_ms"],  # type: ignore[arg-type]
        latency_p95_24h_ms=snapshot["p95_ms"],  # type: ignore[arg-type]
        error_rate_24h=float(snapshot["error_rate"] or 0.0),
        uptime_hours=round(float(snapshot["uptime_hours"] or 0.0), 3),
        inferences_24h=int(snapshot["count"] or 0),
        seed=settings.model_seed,
        checked_at=datetime.now(UTC).isoformat(),
        warnings=warnings,
    )
