"""Resolves the active classifier from configuration."""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings
from app.models.base import Classifier, ImageScreener
from app.models.stub import StubClassifier

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_classifier() -> Classifier:
    """Return the configured backend, loading the model once per process.

    If a backend is requested but cannot be built — a missing checkpoint, missing dependencies, an
    absent API key — the engine falls back to the stub and logs the reason loudly. It does NOT
    pretend the model is live: the stub reports ``operational = False``, so ``/audit/iia-check``
    keeps telling the truth and the thesis cannot accidentally claim IIA = 1.

    The same holds for ``gemini``, which reports ``operational = False`` even when it works: a
    general-purpose multimodal LLM is not the validated CNN the design specifies, so degrading to
    the stub can never turn a true claim into a false one.
    """
    settings = get_settings()

    if settings.model_backend == "torch":
        try:
            if settings.model_arch == "ensemble":
                from app.models.ensemble_backend import EnsembleClassifier, parse_members

                classifier: Classifier = EnsembleClassifier(
                    members=parse_members(settings.ensemble_members),
                    seed=settings.model_seed,
                    input_size=settings.model_input_size,
                    enable_gradcam=settings.enable_gradcam,
                    rule=settings.ensemble_rule,
                )
            else:
                from app.models.torch_backend import TorchClassifier

                classifier = TorchClassifier(
                    checkpoint_path=settings.model_checkpoint_path,
                    seed=settings.model_seed,
                    input_size=settings.model_input_size,
                    enable_gradcam=settings.enable_gradcam,
                    arch=settings.model_arch,
                )
            logger.info("Backend torch activo: %s (%s)", classifier.version, settings.model_arch)
            return classifier
        except Exception as error:
            logger.error(
                "No se pudo cargar el backend torch (%s). Se usa el stub determinista y "
                "/audit/iia-check reportará operational=false.",
                error,
            )

    if settings.model_backend == "gemini":
        try:
            from app.models.gemini_backend import GeminiClassifier

            gemini = GeminiClassifier(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
                timeout_seconds=settings.gemini_timeout_seconds,
                enable_regions=settings.enable_gradcam,
            )
            logger.warning(
                "Backend GEMINI activo (%s). Responde un modelo real, pero NO es la CNN que fija "
                "la seccion 8.1, y /audit/iia-check seguira reportando operational=false.",
                gemini.version,
            )
            return gemini
        except Exception as error:
            logger.error(
                "No se pudo activar el backend gemini (%s). Se usa el stub determinista.",
                error,
            )

    logger.warning(
        "Backend STUB activo: las predicciones son deterministas pero NO provienen de una CNN "
        "entrenada. La tesis no puede declarar IIA = 1 en este estado."
    )
    return StubClassifier(seed=settings.model_seed, enable_gradcam=settings.enable_gradcam)


@lru_cache(maxsize=1)
def get_screener() -> ImageScreener | None:
    """The admission screener, independent of which classifier serves.

    Screening is a different question from grading — "is this a photograph of a face?" versus
    "how severe is the acne?" — and since 17/09/2026 it no longer rides on the classifier: with the
    CNN in service the local screener answers it. ``None`` means nothing screens and the record
    admits the image unchecked, which ``/image/screen`` reports as ``checked: false``.
    """
    settings = get_settings()
    if settings.screening_backend == "none":
        logger.warning("Cribado de admisión desactivado (SCREENING_BACKEND=none).")
        return None
    if settings.screening_backend == "gemini":
        classifier = get_classifier()
        if isinstance(classifier, ImageScreener):
            return classifier
        logger.warning(
            "SCREENING_BACKEND=gemini pero el backend activo (%s) no sabe cribar imágenes; "
            "se usa el cribado local.",
            settings.model_backend,
        )
    try:
        from app.models.local_screener import LocalScreener

        screener = LocalScreener()
        logger.info("Cribado de admisión local activo: %s", screener.version)
        return screener
    except Exception as error:
        logger.error(
            "No se pudo cargar el cribado local (%s). Las imágenes se admiten sin verificar.", error
        )
        return None


def reset_classifier_cache() -> None:
    """Clear the cached classifier and screener; used by the tests."""
    get_classifier.cache_clear()
    get_screener.cache_clear()
