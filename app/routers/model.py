"""Model evaluation over the frozen test set. Covers: OE2 (RF-OE2-04), section 8.5."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from app.config import get_settings
from app.domain.iga import IGA_CLASSES
from app.models.registry import get_classifier
from app.schemas import ModelEvalRequest, ModelEvalResponse, PerClassMetrics

router = APIRouter(prefix="/model", tags=["model"])


@router.post(
    "/eval",
    response_model=ModelEvalResponse,
    summary="Métricas del modelo sobre un split del dataset versionado.",
    description=(
        "Devuelve exactitud, F1 macro, AUC-ROC macro, matriz de confusión y métricas por clase. "
        "Mientras no exista un modelo entrenado y un dataset preparado (fases 2-5 de CRISP-DM), "
        "responde con la estructura vacía y un `note` que explica por qué: un reporte de métricas "
        "que no declara su procedencia es una trampa para el lector."
    ),
)
def model_eval(request: ModelEvalRequest) -> ModelEvalResponse:
    settings = get_settings()
    classifier = get_classifier()

    # A finished training run writes its metrics here; see training/evaluate.py.
    metrics_path = (
        Path(settings.tracking_dir) / request.dataset_version / f"metrics-{request.split}.json"
    )

    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return ModelEvalResponse(
            accuracy=float(payload["accuracy"]),
            f1_macro=float(payload["f1_macro"]),
            auc_roc_macro=payload.get("auc_roc_macro"),
            confusion_matrix=payload["confusion_matrix"],
            per_class_metrics=[PerClassMetrics(**m) for m in payload["per_class_metrics"]],
            dataset_version=request.dataset_version,
            split=request.split,
            model_version=classifier.version,
            run_id=payload.get("run_id"),
            note=None,
        )

    return ModelEvalResponse(
        accuracy=0.0,
        f1_macro=0.0,
        auc_roc_macro=None,
        confusion_matrix=[[0] * len(IGA_CLASSES) for _ in IGA_CLASSES],
        per_class_metrics=[
            PerClassMetrics(
                iga_class=iga_class, support=0, precision=0.0, recall=0.0, specificity=0.0, f1=0.0
            )
            for iga_class in IGA_CLASSES
        ],
        dataset_version=request.dataset_version,
        split=request.split,
        model_version=classifier.version,
        run_id=None,
        note=(
            f"No hay métricas registradas para el dataset {request.dataset_version} "
            f"({request.split}). Ejecute las fases 2-5 de CRISP-DM: preparar el dataset, entrenar "
            f"con training/train.py y evaluar con training/evaluate.py, que escribirá "
            f"{metrics_path}. Backend actual: {classifier.backend}."
        ),
    )
