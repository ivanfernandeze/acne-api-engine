"""Request and response contracts.

These are the shapes section 9.1 of the specification prescribes. ``acne-api`` compiles its client
against the OpenAPI document FastAPI derives from them (``pnpm generate:ai-types``), so this module
is the single definition of the inter-service contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.iga import GagsGrade, IgaClass

# --- Inference ---------------------------------------------------------------


class InferSingleRequest(BaseModel):
    """One image to classify. Bytes travel base64-encoded over the internal REST hop."""

    image_base64: str = Field(..., description="Imagen clínica en base64 (PNG, JPEG o WebP).")
    mime_type: str = Field(default="image/png", description="Tipo MIME de la imagen.")
    case_code: str = Field(default="", description="Código anonimizado del caso, para trazabilidad.")


class Prediction(BaseModel):
    iga_class: IgaClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probs: dict[str, float]


class GagsSecondary(BaseModel):
    grade: GagsGrade
    score: int = Field(..., ge=0, le=44)


class InferSingleResponse(BaseModel):
    prediction: Prediction
    gags_secondary: GagsSecondary
    heatmap_png_base64: str | None = Field(
        default=None, description="Superposición Grad-CAM en PNG base64 (sección 8.4)."
    )
    latency_ms: int
    model_version: str
    model_backend: Literal["stub", "torch"]
    weights_checksum: str | None
    seed: int


# --- Model evaluation --------------------------------------------------------


class ModelEvalRequest(BaseModel):
    dataset_version: str = Field(default="v1.0")
    split: Literal["train", "val", "test"] = Field(default="test")


class PerClassMetrics(BaseModel):
    iga_class: IgaClass
    support: int
    precision: float
    recall: float
    specificity: float
    f1: float


class ModelEvalResponse(BaseModel):
    accuracy: float
    f1_macro: float
    auc_roc_macro: float | None
    confusion_matrix: list[list[int]]
    per_class_metrics: list[PerClassMetrics]
    dataset_version: str
    split: str
    model_version: str
    run_id: str | None
    note: str | None


# --- IIA checklist -----------------------------------------------------------


class IiaCheckResponse(BaseModel):
    """Technical datasheet that validates IIA = 1 (RF-TR-04).

    ``operational`` is false while the engine serves the deterministic stub: the independent
    variable of the thesis is a trained CNN, and the system must say so rather than let the
    absence of a model pass unnoticed.
    """

    operational: bool
    model_backend: Literal["stub", "torch"]
    version: str
    weights_checksum: str | None
    latency_median_24h_ms: float | None
    latency_p95_24h_ms: float | None
    error_rate_24h: float
    uptime_hours: float
    inferences_24h: int
    seed: int
    checked_at: str
    warnings: list[str]


# --- Statistics --------------------------------------------------------------

TestDesign = Literal["paired", "one_sample"]
Alternative = Literal["two-sided", "less", "greater"]
StatisticalTest = Literal["ttest_rel", "wilcoxon", "ttest_1samp", "wilcoxon_1samp"]


class PairedTestRequest(BaseModel):
    design: TestDesign
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    alternative: Alternative = "two-sided"
    pretest: list[float]
    postest: list[float] | None = None
    baseline: float | None = None
    seed: int = 42


class NormalityResult(BaseModel):
    test: Literal["shapiro-wilk"] = "shapiro-wilk"
    statistic: float
    p_value: float
    normal: bool
    applicable: bool
    note: str | None = None


class EffectSize(BaseModel):
    kind: Literal["cohens_dz", "rank_biserial"]
    value: float
    magnitude: Literal["trivial", "small", "medium", "large"]


class PairedTestResponse(BaseModel):
    normality: NormalityResult
    test_applied: StatisticalTest
    statistic: float
    p_value: float
    effect_size: EffectSize | None
    function_version: str


# --- Health ------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    version: str
    model_backend: Literal["stub", "torch"]
    model_operational: bool
    uptime_hours: float
