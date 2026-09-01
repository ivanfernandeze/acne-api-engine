"""Liveness endpoint. Demo requirement of Phase 1 of CRISP-DM."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.models.registry import get_classifier
from app.schemas import HealthResponse
from app.services.telemetry import telemetry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Estado del motor de IA.")
def health() -> HealthResponse:
    classifier = get_classifier()
    snapshot = telemetry.snapshot()

    # The service is up and serving even with the stub; what is "degraded" is the scientific
    # claim, not the process. The distinction is what `operational` in /audit/iia-check carries.
    return HealthResponse(
        status="ok",
        service="acne-ai-engine",
        version=__version__,
        model_backend=classifier.backend,  # type: ignore[arg-type]
        model_operational=classifier.operational,
        uptime_hours=round(float(snapshot["uptime_hours"] or 0.0), 3),
    )
