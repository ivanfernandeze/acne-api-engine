"""Inference endpoint. Covers: OE2 (RF-OE2-01), section 9.1."""

from __future__ import annotations

import base64
import binascii
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypedDict

from fastapi import APIRouter, HTTPException, status

from app.config import get_settings
from app.domain.iga import IGA_TO_GAGS_CENTRE, gags_grade_from_score
from app.models.base import (
    AnamnesisItem,
    ClinicalContext,
    GateResult,
    SecondOpinion,
    SkinConditionChecker,
)
from app.models.registry import get_acne_gate, get_classifier, get_second_opinion
from app.schemas import (
    ClinicalAssessment,
    GagsSecondary,
    InferSingleRequest,
    InferSingleResponse,
    MemberPredictionOut,
    Prediction,
    SecondOpinionOut,
)
from app.services.telemetry import telemetry

logger = logging.getLogger(__name__)


class GateFields(TypedDict):
    """What the acne gate adds to every inference response (see ``InferSingleResponse``)."""

    gate_checked: bool
    acne_probability: float | None
    gate_threshold: float | None
    gate_version: str | None
    gate_probabilities: dict[str, float] | None


router = APIRouter(prefix="/infer", tags=["inference"])

#: One thread for the second opinion, so the remote call overlaps the local grading instead of
#: adding to it. A single worker is enough: the API's queue sends one image at a time.
_second_opinion_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segunda-opinion")


def _second_opinion_result(
    checker: SkinConditionChecker | None, future: Future[SecondOpinion] | None
) -> SecondOpinionOut | None:
    """The advisory answer, or ``None`` if it was off or failed — never an error for the grade."""
    if checker is None or future is None:
        return None
    try:
        opinion = future.result()
    except Exception as error:  # advisory: a failed second opinion must never cost a grade
        logger.warning("La segunda opinión falló (%s). Se gradúa sin ella.", error)
        return None
    return SecondOpinionOut(
        label=opinion.label,
        conditions=list(opinion.conditions),
        explanation=opinion.explanation,
        version=checker.version,
    )

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

    gate_fields = GateFields(
        gate_checked=gate_result is not None,
        acne_probability=None if gate_result is None else gate_result.acne_probability,
        gate_threshold=None if gate_result is None else gate_result.threshold,
        gate_version=None if gate is None or gate_result is None else gate.version,
        gate_probabilities=None if gate_result is None else dict(gate_result.probabilities),
    )

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

    # Only images about to be graded are sent: the gate already explained the rest, and every call
    # is a photograph leaving the machine. It runs alongside the grading, never in front of it.
    checker = get_second_opinion()
    second_future = (
        None
        if checker is None
        else _second_opinion_pool.submit(checker.check, image_bytes, request.mime_type)
    )

    try:
        result = classifier.predict(image_bytes, request.mime_type, context)
    except Exception as error:
        telemetry.record((time.perf_counter() - started) * 1000, failed=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fallo durante la inferencia: {error}",
        ) from error

    second_opinion = _second_opinion_result(checker, second_future)
    latency_ms = (time.perf_counter() - started) * 1000
    telemetry.record(latency_ms)

    # Another condition seen by the second opinion: not graded, exactly like the gate's `no_acne`.
    # The grade was computed in parallel and is discarded here, never returned: a severity of
    # acne for vitiligo must not reach the record. Healthy skin IS on the IGA scale (grade 0), so
    # it is graded and only warned about.
    if (
        second_opinion is not None
        and second_opinion.label == "otra_condicion"
        and settings.second_opinion_stops_grading
    ):
        return InferSingleResponse(
            verdict="no_acne",
            verdict_source="segunda_opinion",
            **gate_fields,
            second_opinion=second_opinion,
            prediction=None,
            gags_secondary=None,
            latency_ms=round(latency_ms),
            model_version=classifier.version,
            model_backend=classifier.backend,  # type: ignore[arg-type]
            weights_checksum=classifier.weights_checksum,
            seed=settings.model_seed,
        )

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
        second_opinion=second_opinion,
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
        latency_ms=round(latency_ms),
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
