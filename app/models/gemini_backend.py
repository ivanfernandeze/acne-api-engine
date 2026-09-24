"""Gemini backend: a multimodal LLM standing in for the CNN.

This is NOT the model the thesis specifies. Section 8.1 names EfficientNet-B2 and section 3 says
"una única CNN validada"; what answers here is a general-purpose multimodal model reached over the
network. It therefore reports ``operational = False`` exactly like the stub — but for a different
reason, which ``/audit/iia-check`` now states explicitly so a reader can tell the two apart.

Two further honesty constraints shape the code:

* **Reproducibility (RNF-03) is not achievable.** ``temperature=0`` is the most determinism the API
  offers, and it is not a guarantee: the remote model version can change without notice and there is
  no seed to fix. The version string carries the model name and the prompt revision so at least the
  configuration that produced a result is recorded.
* **There is no Grad-CAM.** A gradient attribution map needs access to the gradients, which an API
  does not expose. The model is instead asked which regions it considered most affected, and those
  are drawn as boxes. That is the model's own account of what it looked at, not an attribution, and
  the portal labels it as such.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageStat

from app.domain.iga import IGA_CLASSES
from app.models.base import (
    ClassificationResult,
    ClinicalAssessment,
    ClinicalContext,
    ScreeningResult,
)

logger = logging.getLogger(__name__)

#: Bumped whenever the prompt changes. The prompt IS part of the model: a different wording gives
#: different gradings, so a result is only interpretable next to the revision that produced it.
PROMPT_REVISION = "p2"

GAGS_MIN, GAGS_MAX = 0, 44
#: Free text the model writes into the clinical record. Bounded so a runaway answer cannot flood
#: the case file or the PDF report.
MAX_SUMMARY_CHARS = 1200
MAX_ITEM_CHARS = 200
MAX_LIST_ITEMS = 6

SYSTEM_PROMPT = """\
Eres un dermatólogo evaluando acné vulgar. Recibes una fotografía facial y, cuando existe,
la anamnesis del paciente. Emite una lectura clínica razonada, como la dictarías en la consulta.

PRIMERO clasifica la severidad en UNA de las cinco clases de la escala IGA (Investigator's Global
Assessment):

- sin_lesiones: piel limpia, sin lesiones inflamatorias ni comedones.
- casi_sin_lesiones: alguna lesión aislada apenas perceptible.
- leve: pocas lesiones, comedones y algunas pápulas dispersas, sin inflamación marcada.
- moderado: numerosas pápulas y pústulas, inflamación evidente, afectación de varias zonas.
- severo: lesiones muy numerosas, inflamatorias y confluentes, posible presencia de nódulos.

DESPUÉS fundamenta esa graduación:

- summary: tu lectura del caso en prosa, de tres a seis frases. Describe qué ves, en qué zonas y
  con qué grado de inflamación, e integra lo que aporte la anamnesis. Escribe en español, en
  tercera persona y en lenguaje clínico, sin encabezados ni viñetas.
- lesion_types: las lesiones elementales que realmente distingues (comedones cerrados, comedones
  abiertos, pápulas, pústulas, nódulos, quistes, cicatrices, hiperpigmentación postinflamatoria).
- affected_zones: las zonas faciales comprometidas (frente, nariz, mejilla derecha, mejilla
  izquierda, mentón, perioral, mandíbula).
- severity_rationale: una o dos frases sobre POR QUÉ esa clase y no la contigua. La IGA es una
  escala ordinal y la frontera entre dos clases vecinas es donde se juega el acuerdo.
- differentials: otros cuadros que podrían explicar lo que ves (rosácea, foliculitis, dermatitis
  perioral, erupción acneiforme por fármacos). Lista vacía si el cuadro es inequívoco.
- considerations: aspectos de manejo que el especialista debería valorar, en términos generales.
  NUNCA prescribas ni indiques dosis: la decisión terapéutica es del médico.
- anamnesis_flags: lo que la historia clínica cambia y no debe pasarse por alto —posibilidad de
  embarazo, medicación en curso, alergias, enfermedad crónica, tratamientos previos fallidos,
  tendencia a cicatrices o queloides. Deja la lista vacía si no recibes anamnesis.
- gags_score: puntuación GAGS estimada, entero entre 0 y 44.
- confidence: tu confianza en la clase elegida, entre 0 y 1.
- class_probs: la probabilidad de cada una de las cinco clases; deben sumar 1.
- regions: hasta cuatro rectángulos, en coordenadas normalizadas (0 a 1), señalando las zonas más
  afectadas que sustentan tu decisión. Si no hay lesiones, devuelve una lista vacía.

Reglas que no puedes romper:

1. Juzga únicamente lo que ves. No inventes lesiones que no estén presentes ni describas zonas que
   la fotografía no muestre.
2. La anamnesis contextualiza y puede señalar banderas, pero NO decide la graduación: la severidad
   la fija la imagen. Si la historia y la fotografía se contradicen, gradúa por la fotografía y
   dilo en el resumen.
3. Si la imagen no permite evaluar (fuera de foco, mal iluminada, no es un rostro), dilo en
   summary, baja la confianza y no fuerces un diagnóstico.
4. Eres un apoyo diagnóstico. La lectura la valida el especialista.
"""

_STRING_LIST: dict[str, Any] = {"type": "array", "items": {"type": "string"}}

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "iga_class": {"type": "string", "enum": list(IGA_CLASSES)},
        "confidence": {"type": "number"},
        "class_probs": {
            "type": "object",
            "properties": {name: {"type": "number"} for name in IGA_CLASSES},
            "required": list(IGA_CLASSES),
        },
        "gags_score": {"type": "integer"},
        "summary": {"type": "string"},
        "lesion_types": _STRING_LIST,
        "affected_zones": _STRING_LIST,
        "severity_rationale": {"type": "string"},
        "differentials": _STRING_LIST,
        "considerations": _STRING_LIST,
        "anamnesis_flags": _STRING_LIST,
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                },
                "required": ["x", "y", "width", "height"],
            },
        },
    },
    "required": [
        "iga_class",
        "confidence",
        "class_probs",
        "gags_score",
        "summary",
        "lesion_types",
        "affected_zones",
        "severity_rationale",
        "regions",
    ],
}


#: Bumped independently of the grading prompt: the two do different jobs and change separately.
SCREENING_PROMPT_REVISION = "s1"

VALID_SUBJECTS = (
    "rostro_humano",
    "persona_sin_rostro",
    "piel_no_facial",
    "otro",
    "indeterminado",
)

VALID_ISSUES = (
    "encuadre",
    "enfoque",
    "iluminacion",
    "oclusion",
    "resolucion",
    "no_es_fotografia",
)

SCREENING_PROMPT = """\
Eres el control de admisión de un sistema clínico de dermatología. Tu ÚNICA tarea es
decidir si una imagen sirve como fotografía clínica facial. No evalúas la enfermedad.

PROHIBIDO: no cuentes lesiones, no juzgues gravedad, no menciones acné ni ninguna otra dermatosis,
no describas hallazgos clínicos. Si lo haces, la respuesta es inválida.

Clasifica qué muestra la imagen en `subject`:
- rostro_humano: se ve el rostro de una persona real, con la piel facial visible.
- persona_sin_rostro: hay una persona, pero el rostro no se ve o está de espaldas.
- piel_no_facial: piel humana que no es el rostro (espalda, brazo, torso).
- otro: cualquier otra cosa — objeto, animal, paisaje, captura de pantalla, documento, dibujo,
  ilustración, imagen generada, imagen en blanco o sin contenido reconocible.
- indeterminado: ÚNICAMENTE si la imagen es genuinamente ambigua y no puedes decidirte. No la uses
  para decir que no hay un rostro: para eso está `otro`.

`acceptable` es true SOLO si subject = rostro_humano y la piel del rostro se aprecia lo bastante
como para examinarla.

MUY IMPORTANTE: **un rostro sano, limpio y sin ninguna lesión ES admisible**. La ausencia de
lesiones es un hallazgo clínico válido y frecuente en las visitas de control. No rechaces una
imagen por no mostrar enfermedad.

`issues`: problemas de calidad, solo de esta lista y solo si realmente los hay —
encuadre, enfoque, iluminacion, oclusion, resolucion, no_es_fotografia.
Un rostro admisible puede tener problemas de calidad: eso se advierte, no se rechaza.

`reason`: UNA frase en español, dirigida a quien tomó la fotografía, diciéndole qué debe corregir.
Si la imagen es admisible y no tiene problemas, deja la cadena vacía.
"""

SCREENING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subject": {"type": "string", "enum": list(VALID_SUBJECTS)},
        "acceptable": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string", "enum": list(VALID_ISSUES)}},
        "reason": {"type": "string"},
    },
    "required": ["subject", "acceptable", "issues", "reason"],
}


class GeminiResponseError(RuntimeError):
    """The model answered, but not with something that can enter a clinical record."""


class GeminiClassifier:
    """Classifier backed by the Gemini API."""

    backend: str = "gemini"
    #: No local weights: there is nothing to checksum, which is itself part of why RNF-03 fails.
    weights_checksum: str | None = None
    #: A real model answers, but not the CNN the design specifies. See the module docstring.
    operational: bool = False
    #: Distinguishes this from the stub in the IIA checklist: both fail, for different reasons.
    kind: str = "llm_externo"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: int = 45,
        enable_regions: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY no está definida.")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds
        self._enable_regions = enable_regions
        self.version = f"gemini/{model}/{PROMPT_REVISION}"

    def predict(
        self,
        image_bytes: bytes,
        mime_type: str,
        context: ClinicalContext | None = None,
    ) -> ClassificationResult:
        payload = self._call(image_bytes, mime_type, context)
        parsed = self._validate(payload)

        overlay = (
            self._draw_regions(image_bytes, parsed["regions"]) if self._enable_regions else None
        )

        return ClassificationResult(
            iga_class=parsed["iga_class"],
            confidence=parsed["confidence"],
            class_probs=parsed["class_probs"],
            heatmap_png=overlay,
            assessment=ClinicalAssessment(
                summary=parsed["summary"],
                lesion_types=parsed["lesion_types"],
                affected_zones=parsed["affected_zones"],
                severity_rationale=parsed["severity_rationale"],
                differentials=parsed["differentials"],
                considerations=parsed["considerations"],
                anamnesis_flags=parsed["anamnesis_flags"],
                gags_score=parsed["gags_score"],
            ),
        )

    # --- Admission screening -------------------------------------------------

    #: Below this, the image carries essentially no detail: a blank canvas, a flat colour, a
    #: lens cap. Measured as the mean per-channel standard deviation of a downscaled copy.
    BLANK_STDDEV_THRESHOLD = 4.0

    def screen(self, image_bytes: bytes, mime_type: str) -> ScreeningResult:
        """Decide whether the image is an evaluable facial photograph.

        Deliberately a SEPARATE call from `predict`, not a field folded into the grading response.
        Screening happens at intake, which occurs in both experimental conditions; grading happens
        only in the postest. Merging them would either run the grader in the pretest — the exact
        contamination RF-TR-01 exists to prevent — or make admission depend on a call that must not
        happen yet.
        """
        # Cheap, local and certain: an image with no detail cannot be a face, and deciding it here
        # costs nothing, spends no quota and cannot be talked out of by a hedging model. The remote
        # call is reserved for the judgements that actually need one.
        blank = self._is_blank(image_bytes)
        if blank:
            return ScreeningResult(
                acceptable=False,
                subject="otro",
                issues=("no_es_fotografia",),
                reason="La imagen está vacía o no tiene contenido. Tome la fotografía de nuevo.",
            )

        payload = self._ask(
            image_bytes,
            mime_type,
            SCREENING_PROMPT,
            "¿Sirve esta imagen como fotografía clínica facial?",
            SCREENING_SCHEMA,
        )
        return self._validate_screening(payload)

    def _is_blank(self, image_bytes: bytes) -> bool:
        """True when the image carries essentially no detail.

        Returns False on any decoding trouble: this guard exists to catch the obvious, and a
        failure to measure is not evidence of blankness.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                thumbnail = source.convert("RGB")
                thumbnail.thumbnail((64, 64))
                stats = ImageStat.Stat(thumbnail)
                return sum(stats.stddev) / len(stats.stddev) < self.BLANK_STDDEV_THRESHOLD
        except (OSError, ValueError, ZeroDivisionError):
            return False

    def _validate_screening(self, payload: dict[str, Any]) -> ScreeningResult:
        subject = payload.get("subject")
        if subject not in VALID_SUBJECTS:
            raise GeminiResponseError(f'"{subject}" no es un sujeto de imagen reconocido.')

        raw_issues = payload.get("issues")
        issues = tuple(
            issue
            for issue in (raw_issues if isinstance(raw_issues, list) else [])
            if issue in VALID_ISSUES
        )

        # The verdict is derived here, not taken on trust: `acceptable` and `subject` can disagree,
        # and letting a "true" through for something that is not a face would defeat the check.
        # Erring towards acceptance for `indeterminado` is deliberate — a human still looks at every
        # photograph, and blocking a real patient's intake is worse than admitting a doubtful image.
        acceptable = subject == "rostro_humano" and payload.get("acceptable") is True
        # `indeterminado` now means the model genuinely could not decide — the prompt reserves
        # `otro` for "this is plainly not a face". Erring towards acceptance there is deliberate: a
        # human looks at every photograph anyway, and blocking a real patient's intake over a
        # verdict nobody could reach is the worse of the two errors.
        if subject == "indeterminado":
            acceptable = True

        return ScreeningResult(
            acceptable=acceptable,
            subject=subject,
            issues=issues,
            reason=_clean_text(payload.get("reason"), MAX_ITEM_CHARS * 2),
        )

    # --- Transport -----------------------------------------------------------

    def _call(
        self,
        image_bytes: bytes,
        mime_type: str,
        context: ClinicalContext | None = None,
    ) -> dict[str, Any]:
        return self._ask(
            image_bytes, mime_type, SYSTEM_PROMPT, _user_prompt(context), RESPONSE_SCHEMA
        )

    def _ask(
        self,
        image_bytes: bytes,
        mime_type: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """One structured multimodal call. Shared by grading and by admission screening."""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
        )
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            }
                        },
                        {"text": user_prompt},
                    ],
                }
            ],
            "generationConfig": {
                # The most determinism the API offers. Not a guarantee — see the module docstring.
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }

        try:
            response = httpx.post(
                url,
                json=body,
                headers={"x-goog-api-key": self._api_key},
                timeout=self._timeout,
            )
        except httpx.HTTPError as error:
            raise GeminiResponseError(
                f"No se pudo contactar con la API de Gemini: {error}"
            ) from error

        if response.status_code != 200:
            raise GeminiResponseError(
                f"La API de Gemini respondió {response.status_code}: {response.text[:400]}"
            )

        try:
            envelope = response.json()
            text = envelope["candidates"][0]["content"]["parts"][0]["text"]
            parsed: dict[str, Any] = json.loads(text)
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise GeminiResponseError(
                f"Respuesta de Gemini con forma inesperada: {response.text[:400]}"
            ) from error

        return parsed

    # --- Validation ----------------------------------------------------------

    def _validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Reject anything that could not legally enter the clinical record.

        The database enforces CHECK constraints on the IGA class, the confidence range and the GAGS
        score. A value the model invented would fail the INSERT far from here, so it is caught at
        the boundary — and rejected outright rather than coerced: a silently clamped grade is a
        fabricated clinical judgement.
        """
        iga_class = payload.get("iga_class")
        if iga_class not in IGA_CLASSES:
            raise GeminiResponseError(f'"{iga_class}" no es una clase IGA válida.')

        confidence = _as_float(payload.get("confidence"), "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise GeminiResponseError(f"La confianza {confidence} está fuera de [0, 1].")

        raw_probs = payload.get("class_probs")
        if not isinstance(raw_probs, dict):
            raise GeminiResponseError("class_probs no es un objeto.")
        probs = {
            name: _as_float(raw_probs.get(name), f"class_probs.{name}") for name in IGA_CLASSES
        }
        if any(value < 0 for value in probs.values()):
            raise GeminiResponseError("class_probs contiene probabilidades negativas.")

        total = sum(probs.values())
        if total <= 0:
            raise GeminiResponseError("class_probs suma cero.")
        # Renormalising a distribution that is merely off by rounding is arithmetic, not invention;
        # one that is wildly off means the model did not answer the question.
        if not 0.8 <= total <= 1.25:
            raise GeminiResponseError(f"class_probs suma {total:.3f}, demasiado lejos de 1.")
        probs = {name: round(value / total, 4) for name, value in probs.items()}

        gags = payload.get("gags_score")
        if not isinstance(gags, int) or isinstance(gags, bool):
            raise GeminiResponseError(f"gags_score debe ser un entero, llegó {gags!r}.")
        if not GAGS_MIN <= gags <= GAGS_MAX:
            raise GeminiResponseError(f"gags_score {gags} está fuera de [{GAGS_MIN}, {GAGS_MAX}].")

        regions = payload.get("regions")
        clean_regions: list[tuple[float, float, float, float]] = []
        if isinstance(regions, list):
            for region in regions[:4]:
                if not isinstance(region, dict):
                    continue
                try:
                    box = (
                        _as_float(region.get("x"), "x"),
                        _as_float(region.get("y"), "y"),
                        _as_float(region.get("width"), "width"),
                        _as_float(region.get("height"), "height"),
                    )
                except GeminiResponseError:
                    continue
                # Out-of-range boxes are dropped, not clamped: a mislocated box is worse than none.
                if all(0.0 <= value <= 1.0 for value in box) and box[2] > 0 and box[3] > 0:
                    clean_regions.append(box)

        summary = _clean_text(payload.get("summary"), MAX_SUMMARY_CHARS)
        if not summary:
            # The reading in prose is the whole point of asking an LLM instead of a classifier.
            # An empty one means the model answered a different question.
            raise GeminiResponseError("La respuesta no trae lectura clínica (summary vacío).")

        return {
            "iga_class": iga_class,
            "confidence": round(confidence, 4),
            "class_probs": probs,
            "gags_score": gags,
            "regions": clean_regions,
            "summary": summary,
            "lesion_types": _clean_list(payload.get("lesion_types")),
            "affected_zones": _clean_list(payload.get("affected_zones")),
            "severity_rationale": _clean_text(
                payload.get("severity_rationale"), MAX_ITEM_CHARS * 2
            ),
            "differentials": _clean_list(payload.get("differentials")),
            "considerations": _clean_list(payload.get("considerations")),
            "anamnesis_flags": _clean_list(payload.get("anamnesis_flags")),
        }

    # --- Region overlay ------------------------------------------------------

    def _draw_regions(
        self, image_bytes: bytes, regions: list[tuple[float, float, float, float]]
    ) -> bytes | None:
        """Draw the model's declared regions of interest over a dimmed copy of the image.

        Deliberately looks nothing like a heat map: crisp boxes read as "the model says it looked
        here", while a smooth gradient would read as an attribution map the model never produced.
        """
        if not regions:
            return None

        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                canvas = source.convert("RGB")
        except OSError as error:
            logger.warning("No se pudo abrir la imagen para dibujar las regiones: %s", error)
            return None

        width, height = canvas.size
        draw = ImageDraw.Draw(canvas)
        for x, y, box_width, box_height in regions:
            left, top = x * width, y * height
            right = min(width, left + box_width * width)
            bottom = min(height, top + box_height * height)
            draw.rectangle([left, top, right, bottom], outline=(0, 200, 255), width=3)

        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        return buffer.getvalue()


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeminiResponseError(f"{field} debe ser numérico, llegó {value!r}.")
    return float(value)


def _clean_text(value: Any, limit: int) -> str:
    """Collapse a free-text field to a single bounded string, or to ``""`` if it is not one."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.split())
    return text[:limit].rstrip()


def _clean_list(value: Any) -> tuple[str, ...]:
    """Keep the non-empty strings of a list, deduplicated, capped and trimmed.

    Deduplication is case-insensitive because the model repeats itself across fields ("pápulas" in
    lesion_types and again in the rationale), and a repeated bullet reads as two separate findings.
    """
    if not isinstance(value, list):
        return ()
    seen: set[str] = set()
    items: list[str] = []
    for entry in value:
        text = _clean_text(entry, MAX_ITEM_CHARS)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        items.append(text)
        if len(items) == MAX_LIST_ITEMS:
            break
    return tuple(items)


def _user_prompt(context: ClinicalContext | None) -> str:
    """Compose the turn that accompanies the image.

    The anamnesis is quoted verbatim rather than summarised: paraphrasing it here would put a
    second, unversioned interpretation between the record and the model.
    """
    if context is None or context.is_empty():
        return (
            "Evalúa esta fotografía facial. No se dispone de anamnesis: básate solo en la imagen "
            "y deja anamnesis_flags vacío."
        )

    lines = ["Evalúa esta fotografía facial. Datos clínicos del paciente:", ""]
    if context.age is not None:
        lines.append(f"- Edad: {context.age} años")
    if context.sex:
        lines.append(f"- Sexo: {context.sex}")
    if context.phototype:
        lines.append(f"- Fototipo Fitzpatrick: {context.phototype}")
    if context.answers:
        lines.append("")
        lines.append("Anamnesis:")
        lines.extend(f"- {item.question} {item.answer}" for item in context.answers)
    lines.append("")
    lines.append(
        "Recuerda: la severidad la fija la fotografía; la anamnesis contextualiza y señala "
        "banderas."
    )
    return "\n".join(lines)
