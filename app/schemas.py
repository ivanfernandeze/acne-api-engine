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


class AnamnesisItemIn(BaseModel):
    """One answered question of the intake questionnaire, already rendered as text."""

    question: str = Field(..., max_length=300)
    answer: str = Field(..., max_length=600)


class ClinicalContextIn(BaseModel):
    """The history that travels with the photograph.

    Carries no identity: RF-TR-03 reserves the patient's name and document for the treating
    dermatologist, and none of it is clinically necessary to read a photograph. ``acne-api`` builds
    this from `case_anamnesis` and strips everything else.
    """

    age: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = Field(default=None, max_length=40)
    phototype: str | None = Field(default=None, max_length=40)
    answers: list[AnamnesisItemIn] = Field(default_factory=list, max_length=40)


class InferSingleRequest(BaseModel):
    """One image to classify. Bytes travel base64-encoded over the internal REST hop."""

    image_base64: str = Field(..., description="Imagen clínica en base64 (PNG, JPEG o WebP).")
    mime_type: str = Field(default="image/png", description="Tipo MIME de la imagen.")
    case_code: str = Field(
        default="", description="Código anonimizado del caso, para trazabilidad."
    )
    clinical_context: ClinicalContextIn | None = Field(
        default=None,
        description=(
            "Anamnesis anonimizada del caso. Solo la aprovecha un backend capaz de razonar sobre "
            "texto; una CNN la ignora. Nunca contiene identidad del paciente."
        ),
    )


class Prediction(BaseModel):
    iga_class: IgaClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probs: dict[str, float]


class GagsSecondary(BaseModel):
    grade: GagsGrade
    score: int = Field(..., ge=0, le=44)


class ClinicalAssessment(BaseModel):
    """The reasoned reading behind the grade.

    ``None`` for an image-only backend: a CNN emits five numbers and cannot justify them in prose.
    Present when the backend reasons over text, which is what makes the suggestion something the
    physician can accept or reject on its merits instead of on trust.
    """

    summary: str
    lesion_types: list[str] = Field(default_factory=list)
    affected_zones: list[str] = Field(default_factory=list)
    severity_rationale: str = ""
    differentials: list[str] = Field(default_factory=list)
    considerations: list[str] = Field(default_factory=list)
    anamnesis_flags: list[str] = Field(default_factory=list)


class MemberPredictionOut(BaseModel):
    """One model's vote inside the ensemble (section 8.1, revision 17/09/2026)."""

    tag: str
    model_name: str
    model_version: str
    weights_checksum: str
    weight: float = Field(..., ge=0.0, description="Peso en la votación: F1 macro en el test.")
    iga_class: IgaClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_probs: dict[str, float]
    heatmap_png_base64: str | None = None
    latency_ms: int


class InferSingleResponse(BaseModel):
    prediction: Prediction
    gags_secondary: GagsSecondary
    assessment: ClinicalAssessment | None = Field(
        default=None,
        description="Lectura clínica razonada. Ausente cuando el backend solo mira píxeles.",
    )
    heatmap_png_base64: str | None = Field(
        default=None, description="Superposición Grad-CAM en PNG base64 (sección 8.4)."
    )
    latency_ms: int
    model_version: str
    model_backend: Literal["stub", "torch", "gemini"]
    weights_checksum: str | None
    seed: int
    members: list[MemberPredictionOut] = Field(
        default_factory=list,
        description=(
            "Lecturas individuales cuando el motor sirve un ensemble; `prediction` es la "
            "recomendación combinada según ENSEMBLE_RULE (por defecto, la lectura íntegra del "
            "miembro más seguro; empate → mayor peso). Vacío con un solo modelo."
        ),
    )
    agreement: Literal["unanime", "mayoria", "discrepancia"] | None = Field(
        default=None,
        description="Acuerdo entre los miembros del ensemble; null con un solo modelo.",
    )


# --- Admission screening -----------------------------------------------------


class ImageScreenRequest(BaseModel):
    """One image to judge for admissibility, before it enters the clinical record."""

    image_base64: str = Field(..., description="Imagen candidata en base64 (PNG, JPEG o WebP).")
    mime_type: str = Field(default="image/png", description="Tipo MIME de la imagen.")


class ImageScreenResponse(BaseModel):
    """Whether an image may be filed as a clinical photograph.

    THERE IS DELIBERATELY NO CLINICAL FIELD HERE — no severity, no lesion count, no IGA class. The
    check runs at intake, which happens in BOTH experimental conditions, so anything it revealed
    about the disease would put the intervention into the pretest and break RF-TR-01. Being
    identical on both sides is precisely what makes the check admissible.
    """

    acceptable: bool = Field(..., description="Falso solo si la imagen no debe entrar al registro.")
    checked: bool = Field(
        ...,
        description=(
            "Falso cuando el backend activo no sabe examinar imágenes (stub, CNN). En ese caso "
            "`acceptable` es true por defecto y no significa que se haya verificado nada."
        ),
    )
    subject: str = Field(..., description="Qué muestra la imagen; solo admisibilidad.")
    issues: list[str] = Field(
        default_factory=list,
        description="Problemas de calidad. Advierten; por sí solos no bloquean la subida.",
    )
    reason: str = Field(default="", description="Una frase para quien tomó la fotografía.")
    latency_ms: int
    model_backend: Literal["stub", "torch", "gemini"]


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

    ``operational`` is false while the engine serves anything other than a trained CNN: the
    independent variable of the thesis is that CNN, and the system must say so rather than let its
    absence pass unnoticed.

    ``model_kind`` says WHY it is false, which ``operational`` alone cannot. A stub and an external
    LLM both fail the check, for entirely different reasons, and a reader of the checklist has to be
    able to tell them apart.
    """

    operational: bool
    model_backend: Literal["stub", "torch", "gemini"]
    model_kind: Literal["stub", "cnn", "llm_externo"]
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
    ensemble_members: list[EnsembleMemberInfo] = Field(
        default_factory=list,
        description="Modelos que componen el ensemble servido, con versión y checksum de cada uno.",
    )


class EnsembleMemberInfo(BaseModel):
    tag: str
    model_name: str
    version: str
    weights_checksum: str
    weight: float


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
    model_backend: Literal["stub", "torch", "gemini"]
    model_operational: bool
    uptime_hours: float
