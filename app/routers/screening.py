"""Admission screening: is this image a clinical facial photograph at all?

EXTENSION requested by the research team; no numbered requirement of `requisitos.md` backs it, and
it is declared in `acne-infra/TRACEABILITY.md`.

Two properties make it safe to run inside the experiment:

* **It says nothing clinical.** The response has no field for severity, lesion count or IGA class,
  and the prompt forbids mentioning them. Screening happens at intake, which occurs in BOTH the
  pretest and the postest condition; anything it revealed about the disease would put the
  intervention into the pretest and break RF-TR-01.
* **It is identical in both conditions.** A factor held constant on both sides of a paired contrast
  cannot explain a difference between them, so it cannot bias TRM, TPD or TCAD.

It is also deliberately permissive about the disease: a clean, healthy face is admissible.
``sin_lesiones`` is one of the five IGA classes, and a resolved case at a follow-up visit is exactly
what it looks like — screening for "shows acne" would reject the evidence that treatment worked.
"""

from __future__ import annotations

import base64
import binascii
import logging
import time

from fastapi import APIRouter, HTTPException, status

from app.models.registry import get_classifier, get_screener
from app.schemas import ImageScreenRequest, ImageScreenResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/image", tags=["screening"])

MAX_IMAGE_BYTES = 15 * 1024 * 1024


@router.post(
    "/screen",
    response_model=ImageScreenResponse,
    summary="Decide si una imagen es una fotografía clínica facial admisible.",
    description=(
        "No emite ningún juicio clínico: no menciona lesiones, gravedad ni clase IGA. Se ejecuta "
        "en la admisión, que ocurre en las dos condiciones del experimento, y por eso no puede "
        "revelar nada de la enfermedad (RF-TR-01). Un rostro sano y sin lesiones ES admisible."
    ),
)
def screen_image(request: ImageScreenRequest) -> ImageScreenResponse:
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
    screener = get_screener()
    started = time.perf_counter()

    # Nothing configured to examine images says so instead of inventing a verdict. The caller then
    # admits the image, exactly as it did before this endpoint existed — `checked` is what tells
    # it apart from a real pass.
    if screener is None:
        return ImageScreenResponse(
            acceptable=True,
            checked=False,
            subject="indeterminado",
            issues=[],
            reason="",
            latency_ms=0,
            model_backend=classifier.backend,  # type: ignore[arg-type]
        )

    try:
        result = screener.screen(image_bytes, request.mime_type)
    except Exception as error:  # never block an intake on our own failure
        # Failing open is the deliberate choice: a physician looks at every photograph anyway, and
        # refusing a real patient's intake because a remote API blinked is a worse outcome than
        # admitting an image nobody screened. The log is loud so the silence is not mistaken for a
        # verdict.
        logger.error("El cribado de admisión falló (%s). La imagen se admite sin verificar.", error)
        return ImageScreenResponse(
            acceptable=True,
            checked=False,
            subject="indeterminado",
            issues=[],
            reason="",
            latency_ms=round((time.perf_counter() - started) * 1000),
            model_backend=classifier.backend,  # type: ignore[arg-type]
        )

    return ImageScreenResponse(
        acceptable=result.acceptable,
        checked=True,
        subject=result.subject,
        issues=list(result.issues),
        reason=result.reason,
        pose=result.pose,
        yaw_ratio=result.yaw_ratio,
        latency_ms=round((time.perf_counter() - started) * 1000),
        model_backend=classifier.backend,  # type: ignore[arg-type]
    )
