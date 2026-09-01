"""Resolves the active classifier from configuration."""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings
from app.models.base import Classifier
from app.models.stub import StubClassifier

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_classifier() -> Classifier:
    """Return the configured backend, loading the model once per process.

    If ``MODEL_BACKEND=torch`` is requested but the checkpoint or the dependencies are missing,
    the engine falls back to the stub and logs the reason loudly. It does NOT pretend the model
    is live: the stub reports ``operational = False``, so ``/audit/iia-check`` keeps telling the
    truth and the thesis cannot accidentally claim IIA = 1.
    """
    settings = get_settings()

    if settings.model_backend == "torch":
        try:
            from app.models.torch_backend import TorchClassifier

            classifier = TorchClassifier(
                checkpoint_path=settings.model_checkpoint_path,
                seed=settings.model_seed,
                input_size=settings.model_input_size,
                enable_gradcam=settings.enable_gradcam,
            )
            logger.info("Backend torch activo: %s", classifier.version)
            return classifier
        except Exception as error:  # noqa: BLE001 - surface any load failure, keep serving
            logger.error(
                "No se pudo cargar el backend torch (%s). Se usa el stub determinista y "
                "/audit/iia-check reportará operational=false.",
                error,
            )

    logger.warning(
        "Backend STUB activo: las predicciones son deterministas pero NO provienen de una CNN "
        "entrenada. La tesis no puede declarar IIA = 1 en este estado."
    )
    return StubClassifier(seed=settings.model_seed, enable_gradcam=settings.enable_gradcam)


def reset_classifier_cache() -> None:
    """Clear the cached classifier; used by the tests."""
    get_classifier.cache_clear()
