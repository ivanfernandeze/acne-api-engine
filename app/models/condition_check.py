"""Second opinion before grading: does this photograph show acne, healthy skin or something else?

Why it exists. The local acne gate (``acne_gate.py``) was trained on ACNE04 close-ups for acne,
studio portraits (CFD) for healthy skin and Fitzpatrick17k for every other condition, and part of
what it learned is the camera: a phone close-up of a cheek looks like its «acne» class whatever the
skin shows. A phone photograph of vitiligo with freckles came out ``p(acne) = 0.995``, and no
threshold stops that. A general multimodal model has seen far more dermatology than those three
datasets and is a cheap independent check.

What it may and may not do:

* **Stops the grade only for another condition.** When it sees vitiligo, melasma… the image is not
  graded (``verdict = no_acne``, ``verdict_source = segunda_opinion``), like the gate's own
  ``no_acne``: an IGA grade for vitiligo means nothing. Healthy skin or acne next to something else
  is graded with a warning. A failure, no key or a timeout never costs a grade. The specialist
  always decides. ``SECOND_OPINION_STOPS_GRADING=false`` makes it purely advisory.
* **Only where inference runs.** It is consulted from ``/infer/single`` and never from
  ``/image/screen``: inference is not invoked in the pretest, so a «this looks like vitiligo» cannot
  reach that condition (RF-TR-01).
* **The photograph leaves this machine.** It is sent to Google's API. To keep that to the minimum it
  is re-encoded before sending — longest side at most ``MAX_SIDE`` pixels, JPEG, no EXIF (so no GPS,
  device or timestamp) — and nothing else goes with it: no name, no code, no anamnesis. The study's
  informed consent has to say that an external service analyses the image, and the key must belong
  to a paid tier, whose terms exclude using the content to train Google's models.
* **Not reproducible** (RNF-03): the remote model can change without notice. The version string
  carries the model and the prompt revision, and none of the thesis metrics depend on it.
"""

from __future__ import annotations

import io
import logging
from typing import Any

from PIL import Image, ImageOps

from app.models.base import SecondOpinion, SecondOpinionLabel
from app.models.gemini_backend import GeminiClassifier, GeminiResponseError

logger = logging.getLogger(__name__)

#: Bumped whenever the prompt changes: an answer is only interpretable next to the prompt that
#: produced it.
CONDITION_PROMPT_REVISION = "c1"

#: Longest side, in pixels, of the copy that is sent. Enough to tell vitiligo from acne; far less
#: than the original photograph.
MAX_SIDE = 1024
JPEG_QUALITY = 88

MAX_CONDITIONS = 4
MAX_CONDITION_CHARS = 60
MAX_EXPLANATION_CHARS = 300

VALID_LABELS: tuple[SecondOpinionLabel, ...] = (
    "acne",
    "piel_sana",
    "otra_condicion",
    "indeterminado",
)

CONDITION_PROMPT = """\
Eres un dermatólogo que hace un TRIAJE PREVIO de una fotografía de piel facial, antes de que otro
sistema gradúe la severidad del acné. Tu ÚNICA tarea es decir qué muestra la imagen.

Clasifica en `label`:
- acne: acné vulgar ACTIVO — comedones abiertos o cerrados, pápulas o pústulas inflamatorias,
  nódulos o quistes.
- piel_sana: piel sin acné activo ni otra enfermedad. Las pecas (efélides), los lunares, los poros
  visibles, el brillo, la textura normal y las cicatrices antiguas planas son PIEL SANA.
- otra_condicion: la imagen muestra principalmente otra condición y no acné activo — por ejemplo
  vitíligo (máculas blancas despigmentadas), melasma, rosácea, dermatitis seborreica o perioral,
  psoriasis, foliculitis, milia, queratosis pilar, verrugas, lupus, tiña, hiperpigmentación
  postinflamatoria sin lesiones activas.
- indeterminado: SOLO si la imagen no permite decidir (borrosa, muy oscura, la piel no se ve).

`conditions`: TODAS las condiciones distintas del acné que se observan, en español, con su nombre
clínico corto ("vitíligo", "melasma", "rosácea"…), la más evidente primero. Inclúyelas aunque
`label` sea acne: el profesional necesita saber que el acné coexiste con otra cosa. NO incluyas
pecas, lunares ni poros: son piel sana. Lista vacía si no hay ninguna.

`explanation`: UNA frase en español para el dermatólogo, describiendo lo que ves que justifica la
clasificación (tipo de lesión, color, bordes, distribución).

PROHIBIDO: no gradúes la severidad del acné, no cuentes lesiones, no propongas tratamiento.
"""

CONDITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(VALID_LABELS)},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": ["label", "conditions", "explanation"],
}


def prepare_for_upload(image_bytes: bytes) -> tuple[bytes, str]:
    """The copy that leaves the machine: upright, at most ``MAX_SIDE`` px, JPEG, no metadata.

    Re-encoding from pixels drops every EXIF field — GPS position, device, capture time — which the
    patient never agreed to share with anyone. ``exif_transpose`` first, so a phone photograph that
    relies on its orientation tag is not sent sideways.
    """
    with Image.open(io.BytesIO(image_bytes)) as source:
        upright = ImageOps.exif_transpose(source).convert("RGB")
    upright.thumbnail((MAX_SIDE, MAX_SIDE))
    buffer = io.BytesIO()
    upright.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue(), "image/jpeg"


def parse_opinion(payload: dict[str, Any]) -> SecondOpinion:
    """Turn the model's JSON into a ``SecondOpinion``, refusing anything off-contract.

    A label outside the four is an error, not something to coerce: a guessed «acne» would hide the
    very warning this exists to give.
    """
    label = payload.get("label")
    if label not in VALID_LABELS:
        raise GeminiResponseError(f'"{label}" no es una clasificación reconocida.')

    raw = payload.get("conditions")
    conditions: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())[:MAX_CONDITION_CHARS].strip()
        # Freckles are healthy skin; the prompt says so, and this keeps a model that forgets from
        # raising a warning about them.
        if text and text.lower() not in {c.lower() for c in conditions} and not _is_benign(text):
            conditions.append(text)
        if len(conditions) == MAX_CONDITIONS:
            break

    explanation = payload.get("explanation")
    return SecondOpinion(
        label=label,
        conditions=tuple(conditions),
        explanation=(
            " ".join(explanation.split())[:MAX_EXPLANATION_CHARS]
            if isinstance(explanation, str)
            else ""
        ),
    )


def _is_benign(condition: str) -> bool:
    lowered = condition.lower()
    return any(word in lowered for word in ("peca", "efélide", "efelide", "lunar", "poro"))


class GeminiConditionChecker:
    """``SkinConditionChecker`` backed by the Gemini API, reusing its transport."""

    def __init__(self, api_key: str, model: str, timeout_seconds: int = 20) -> None:
        # The classifier is only the HTTP client here: no grading prompt is ever sent through it.
        self._client = GeminiClassifier(
            api_key=api_key, model=model, timeout_seconds=timeout_seconds, enable_regions=False
        )
        self.version = f"gemini/{model}/{CONDITION_PROMPT_REVISION}"

    def check(self, image_bytes: bytes, mime_type: str) -> SecondOpinion:
        try:
            payload, payload_mime = prepare_for_upload(image_bytes)
        except (OSError, ValueError) as error:
            raise GeminiResponseError(f"No se pudo preparar la imagen: {error}") from error
        # Same package: the transport is shared, so a fix to it reaches both uses.
        answer = self._client._ask(
            payload,
            payload_mime,
            CONDITION_PROMPT,
            "¿Esta fotografía muestra acné, piel sana u otra condición?",
            CONDITION_SCHEMA,
        )
        return parse_opinion(answer)
