"""Classifier interface shared by the stub and the real CNN backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from app.domain.iga import IgaClass


@dataclass(frozen=True)
class AnamnesisItem:
    """One answered question of the intake questionnaire, already rendered as text."""

    question: str
    answer: str


@dataclass(frozen=True)
class ClinicalContext:
    """The history that accompanies the photograph, anonymised.

    Carries NO identity: no name, no document, no patient code. A backend that reasons over text
    needs the clinical facts — how long the lesions have been there, whether they leave scars,
    whether the patient could be pregnant — and needs none of the identity that RF-TR-03 reserves
    for the treating dermatologist.
    """

    age: int | None = None
    sex: str | None = None
    phototype: str | None = None
    answers: tuple[AnamnesisItem, ...] = ()

    def is_empty(self) -> bool:
        return self.age is None and self.sex is None and self.phototype is None and not self.answers


@dataclass(frozen=True)
class ClinicalAssessment:
    """The reasoned reading behind a grade.

    A bare class plus a probability vector tells the physician *what* the model concluded and
    nothing about *why*, which is precisely what makes a suggestion impossible to accept or reject
    on its merits. Only a backend that can reason over text produces this; a CNN outputs five
    numbers and leaves this ``None``.
    """

    #: The reading in prose, as a dermatologist would dictate it.
    summary: str
    #: Elementary lesions actually seen: comedones, pápulas, pústulas, nódulos, cicatrices…
    lesion_types: tuple[str, ...] = ()
    #: Facial zones involved (frente, mejillas, mentón…), which is what GAGS scores by region.
    affected_zones: tuple[str, ...] = ()
    #: Why THAT IGA class and not the neighbouring one. The ordinal scale lives or dies here.
    severity_rationale: str = ""
    #: What else could look like this: rosácea, foliculitis, dermatitis perioral…
    differentials: tuple[str, ...] = ()
    #: Management points for the physician to weigh. Never a prescription and never a dose.
    considerations: tuple[str, ...] = ()
    #: What the history changes: pregnancy, current medication, allergies, prior failed treatment.
    anamnesis_flags: tuple[str, ...] = ()
    #: GAGS estimated by the model itself, when it can. ``None`` falls back to the IGA band centre.
    gags_score: int | None = None


#: What the photograph actually shows. Admissibility only — nothing about the disease.
ImageSubject = Literal[
    "rostro_humano",
    "persona_sin_rostro",
    "piel_no_facial",
    "otro",
    "indeterminado",
]


@dataclass(frozen=True)
class ScreeningResult:
    """Whether an image may enter the clinical record at all.

    THIS IS NOT A DIAGNOSIS, and the type is shaped so it cannot become one: there is no field for
    severity, lesion count or IGA class, and the prompt forbids mentioning them. That constraint is
    load-bearing for the experiment — the check runs at intake, which happens in BOTH the pretest
    and the postest condition, so anything it revealed about the disease would leak the very
    intervention that RF-TR-01 keeps out of the pretest.

    Being constant across both conditions is also why the check is admissible at all: a factor held
    identical on both sides cannot explain a difference between them.

    A HEALTHY FACE IS ADMISSIBLE. ``sin_lesiones`` is one of the five IGA classes and a clean face
    at a follow-up visit is exactly what a resolved case looks like. Screening for "shows acne"
    would reject the evidence that the treatment worked.
    """

    acceptable: bool
    subject: ImageSubject
    #: Quality problems worth telling the photographer about: encuadre, enfoque, iluminación…
    #: These WARN; they never block on their own.
    issues: tuple[str, ...] = ()
    #: One sentence addressed to whoever took the photograph, saying what to fix.
    reason: str = ""


@runtime_checkable
class ImageScreener(Protocol):
    """Optional capability: deciding whether an image is an evaluable facial photograph.

    Separate from ``Classifier`` because only a backend that understands images semantically can do
    it. A CNN trained to output five IGA logits has no opinion on whether it was handed a landscape,
    and the honest answer for it is "not checked" rather than a fabricated verdict.
    """

    def screen(self, image_bytes: bytes, mime_type: str) -> ScreeningResult:
        """Decide whether the image may enter the clinical record."""
        ...


Agreement = Literal["unanime", "mayoria", "discrepancia"]


@dataclass(frozen=True)
class MemberPrediction:
    """One model's reading inside an ensemble, kept whole so the physician can see each vote."""

    tag: str
    model_name: str
    model_version: str
    weights_checksum: str
    #: Weight in the soft vote — the model's F1 macro on the frozen test split.
    weight: float
    iga_class: IgaClass
    confidence: float
    class_probs: dict[str, float]
    heatmap_png: bytes | None
    latency_ms: int


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of one inference, independent of which backend produced it."""

    iga_class: IgaClass
    confidence: float
    class_probs: dict[str, float]
    #: Grad-CAM overlay as PNG bytes; ``None`` when interpretability is disabled.
    heatmap_png: bytes | None
    #: Present only for a backend that reasons over text. See ``ClinicalAssessment``.
    assessment: ClinicalAssessment | None = field(default=None)
    #: The individual readings when several models were run together; empty for a single model.
    members: tuple[MemberPrediction, ...] = field(default=())
    #: How much the members agreed: all the same class, two of three, or none.
    agreement: Agreement | None = field(default=None)


class Classifier(Protocol):
    """Contract every backend must satisfy.

    Keeping the stub and the CNN behind one protocol is what lets the whole clinical pipeline —
    queue, Grad-CAM display, TPD, confusion matrix, HE2 contrast — be built and verified before a
    single epoch of training has run, and then switched over without touching a caller.
    """

    #: Reported in every inference and in the IIA checklist.
    version: str
    backend: str
    #: SHA-256 of the weights file; ``None`` when there are no weights.
    weights_checksum: str | None
    #: True only when a genuinely trained model is serving, which is what gates IIA = 1.
    operational: bool
    #: Why the check passes or fails: ``stub``, ``cnn`` or ``llm_externo``. Two backends can both
    #: report ``operational = False`` for reasons that are not remotely equivalent.
    kind: str

    def predict(
        self,
        image_bytes: bytes,
        mime_type: str,
        context: ClinicalContext | None = None,
    ) -> ClassificationResult:
        """Classify one image into the five IGA classes.

        ``context`` is the anamnesis. An image-only model ignores it — and must keep accepting it,
        so that swapping backends never changes the caller.
        """
        ...
