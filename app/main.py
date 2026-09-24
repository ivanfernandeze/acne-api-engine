"""FastAPI application of the acne AI engine.

Prototype supporting the thesis "Agilizacion del proceso de diagnostico de acne mediante
inteligencia artificial en un centro especializado en dermatologia" (UNT, 2026).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import get_settings
from app.models.registry import get_classifier
from app.routers import audit, health, infer, model, screening, stats

DISCLAIMER = (
    "Herramienta de apoyo diagnóstico con fines de investigación académica. No sustituye el "
    "juicio clínico del profesional médico. Prototipo académico UNT — no certificado como "
    "dispositivo médico. Diagnóstico preliminar sujeto a validación por el especialista."
)

DESCRIPTION = f"""
Motor de inferencia CNN, interpretabilidad Grad-CAM y análisis estadístico del prototipo de
diagnóstico asistido de acné.

> {DISCLAIMER}

### Objetivos que cubre

| Endpoint | Objetivo | Requerimiento |
|---|---|---|
| `POST /infer/single` | OE2 — precisión del diagnóstico preliminar | RF-OE2-01 |
| `POST /model/eval` | OE2 — métricas sobre el test congelado | RF-OE2-04 |
| `GET /audit/iia-check` | Variable independiente (IIA) | RF-TR-04 |
| `POST /stats/paired-test` | HE₁–HE₄ | RF-OE1-04, RF-OE2-05, RF-OE3-05, RF-OE4-03 |

### Regla de decisión estadística

Shapiro-Wilk sobre las diferencias pareadas → t de Student para muestras relacionadas si no se
rechaza la normalidad, o prueba de rangos con signo de Wilcoxon en caso contrario. α = 0,05.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the classifier at startup so the first clinical request does not pay for it."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    logger = logging.getLogger("acne-ai-engine")

    classifier = get_classifier()
    logger.info(
        "acne-ai-engine %s listo · backend=%s · versión del modelo=%s · operativo=%s",
        __version__,
        classifier.backend,
        classifier.version,
        classifier.operational,
    )
    if not classifier.operational:
        logger.warning(
            "IIA = 0: no hay una CNN entrenada activa. /audit/iia-check reportará "
            "operational=false y la matriz de cobertura marcará OE2 y OG como 'partial'."
        )
    yield


app = FastAPI(
    title="acne-ai-engine",
    description=DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# The orchestrator is the only intended client, but a permissive CORS policy lets a developer
# poke the engine directly from the portal's browser tab during a demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(infer.router)
app.include_router(screening.router)
app.include_router(model.router)
app.include_router(audit.router)
app.include_router(stats.router)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "acne-ai-engine",
        "version": __version__,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "disclaimer": DISCLAIMER,
    }
