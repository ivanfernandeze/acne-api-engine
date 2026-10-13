"""Inference endpoint. Covers: OE2 (RF-OE2-01), section 9.1."""

from __future__ import annotations

import base64
import binascii
import logging
import time

from fastapi import APIRouter, HTTPException, status

from app.config import get_settings
from app.domain.iga import IGA_TO_GAGS_CENTRE, gags_grade_from_score
from app.models.base import AnamnesisItem, ClinicalContext, GateResult
from app.models.registry import get_acne_gate, get_classifier
from app.schemas import (
    ClinicalAssessment,
    GagsSecondary,
    InferSingleRequest,
    InferSingleResponse,
    MemberPredictionOut,
    Prediction,
)
from app.services.telemetry import telemetry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/infer", tags=["inference"])

MAX_IMAGE_BYTES = 15 * 1024 * 1024


@router.post(
    "/single",
    response_model=InferSingleResponse,
    summary="Clasifica una imagen facial en la escala IGA y devuelve el Grad-CAM.",
)
def infer_single(request: InferSingleRequest) -> InferSingleResponse:
    """Run one inference.

    The measured latency is the engine's own processing time; ``acne-api`` separately records the
    end-to-end latency, and ``/audit/iia-check`` reports both. The distinction matters for RNF-01:
    the budget is about what the clinician waits for, not about what the model alone costs.
    """
    settings = get_settings()

    try:
        image_bytes = base64.b64decode(request.image_base64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"La imagen no está correctamente codificada en base64: {error}",
        ) from error

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="La imagen recibida está vacía.",
        )
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"La imagen supera el máximo de {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
        )

    classifier = get_classifier()
    started = time.perf_counter()

    # The gate first: an image that is not acne is not graded at all. Grading it anyway and hiding
    # the grade would still put a severity for vitiligo or freckles into the record.
    gate = get_acne_gate()
    gate_result: GateResult | None = None
    if gate is not None:
        try:
            gate_result = gate.check(image_bytes, request.mime_type)
        except Exception as error:  # the gate must never cost a diagnosis
            # Failing open, like the screener: a clinician reviews every case anyway.
            logger.error("El filtro de acné falló (%s). La imagen se gradúa sin filtrar.", error)

    gate_fields = {
        "gate_checked": gate_result is not None,
        "acne_probability": None if gate_result is None else gate_result.acne_probability,
        "gate_threshold": None if gate_result is None else gate_result.threshold,
        "gate_version": None if gate is None or gate_result is None else gate.version,
        "gate_probabilities": None if gate_result is None else dict(gate_result.probabilities),
    }

    if gate_result is not None and not gate_result.is_acne:
        latency_ms = (time.perf_counter() - started) * 1000
        telemetry.record(latency_ms)
        healthy = gate_result.label == "sana"
        # Healthy skin IS a reading: `sin_lesiones`, the class ACNE04 never let the IGA models
        # learn. Its confidence is the gate's, and the vector says so by carrying only that class.
        # Another condition has no IGA reading at all.
        healthy_probability = gate_result.probabilities.get("sana", 0.0)
        return InferSingleResponse(
            verdict="piel_sana" if healthy else "no_acne",
            **gate_fields,
            prediction=(
                Prediction(
                    iga_class="sin_lesiones",
                    confidence=healthy_probability,
                    class_probs={"sin_lesiones": healthy_probability},
                )
                if healthy
                else None
            ),
            gags_secondary=(
                GagsSecondary(
                    grade=gags_grade_from_score(IGA_TO_GAGS_CENTRE["sin_lesiones"]),
                    score=IGA_TO_GAGS_CENTRE["sin_lesiones"],
                )
                if healthy
                else None
            ),
            latency_ms=round(latency_ms),
            model_version=classifier.version,
            model_backend=classifier.backend,  # type: ignore[arg-type]
            weights_checksum=classifier.weights_checksum,
            seed=settings.model_seed,
        )

    context = (
        None
        if request.clinical_context is None
        else ClinicalContext(
            age=request.clinical_context.age,
            sex=request.clinical_context.sex,
            phototype=request.clinical_context.phototype,
            answers=tuple(
                AnamnesisItem(question=item.question, answer=item.answer)
                for item in request.clinical_context.answers
            ),
        )
    )

    try:
        result = classifier.predict(image_bytes, request.mime_type, context)
    except Exception as error:  # noqa: BLE001 - report the failure rate honestly
        telemetry.record((time.perf_counter() - started) * 1000, failed=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fallo durante la inferencia: {error}",
        ) from error

    latency_ms = (time.perf_counter() - started) * 1000
    telemetry.record(latency_ms)

    # Prefer the model's own GAGS when it produced one. The band centre is a presentational
    # fallback derived from the IGA class, so using it over a real estimate would throw away the
    # only genuine reading of the secondary scale the system ever gets.
    assessment = result.assessment
    gags_score = (
        assessment.gags_score
        if assessment is not None and assessment.gags_score is not None
        else IGA_TO_GAGS_CENTRE[result.iga_class]
    )

    return InferSingleResponse(
        verdict="acne",
        **gate_fields,
        prediction=Prediction(
            iga_class=result.iga_class,
            confidence=result.confidence,
            class_probs=result.class_probs,
        ),
        gags_secondary=GagsSecondary(
            grade=gags_grade_from_score(gags_score),
            score=gags_score,
        ),
        assessment=(
            None
            if assessment is None
            else ClinicalAssessment(
                summary=assessment.summary,
                lesion_types=list(assessment.lesion_types),
                affected_zones=list(assessment.affected_zones),
                severity_rationale=assessment.severity_rationale,
                differentials=list(assessment.differentials),
                considerations=list(assessment.considerations),
                anamnesis_flags=list(assessment.anamnesis_flags),
            )
        ),
        heatmap_png_base64=(
            base64.b64encode(result.heatmap_png).decode("ascii")
            if result.heatmap_png is not None
            else None
        ),
        latency_ms=int(round(latency_ms)),
        model_version=classifier.version,
        model_backend=classifier.backend,  # type: ignore[arg-type]
        weights_checksum=classifier.weights_checksum,
        seed=settings.model_seed,
        members=[
            MemberPredictionOut(
                tag=member.tag,
                model_name=member.model_name,
                model_version=member.model_version,
                weights_checksum=member.weights_checksum,
                weight=member.weight,
                iga_class=member.iga_class,
                confidence=member.confidence,
                class_probs=member.class_probs,
                heatmap_png_base64=(
                    base64.b64encode(member.heatmap_png).decode("ascii")
                    if member.heatmap_png is not None
                    else None
                ),
                latency_ms=member.latency_ms,
            )
            for member in result.members
        ],
        agreement=result.agreement,
    )
