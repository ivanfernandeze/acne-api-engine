"""Validation tests for the Gemini backend.

The model is asked for a clinical grade and answers with free-formed JSON. Everything it returns
crosses into a clinical record protected by database CHECK constraints, so the boundary has to be
strict: a value that is merely coerced into range would be a fabricated clinical judgement, and one
that slips through fails the INSERT far from where it was produced.

None of these tests call the API.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from app.models.base import AnamnesisItem, ClinicalContext
from app.models.gemini_backend import GeminiClassifier, GeminiResponseError, _user_prompt


@pytest.fixture
def classifier() -> GeminiClassifier:
    return GeminiClassifier(api_key="clave-de-prueba", model="gemini-2.5-flash")


def valid_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "iga_class": "moderado",
        "confidence": 0.78,
        "class_probs": {
            "sin_lesiones": 0.02,
            "casi_sin_lesiones": 0.05,
            "leve": 0.10,
            "moderado": 0.78,
            "severo": 0.05,
        },
        "gags_score": 22,
        "regions": [{"x": 0.2, "y": 0.3, "width": 0.15, "height": 0.12}],
        "summary": (
            "Se observan pápulas y pústulas inflamatorias en ambas mejillas y mentón, con "
            "afectación moderada y algunas lesiones residuales hiperpigmentadas."
        ),
        "lesion_types": ["pápulas", "pústulas", "comedones cerrados"],
        "affected_zones": ["mejilla derecha", "mejilla izquierda", "mentón"],
        "severity_rationale": "Supera el umbral de leve por el número de lesiones inflamatorias.",
        "differentials": ["rosácea papulopustulosa"],
        "considerations": ["Valorar terapia tópica combinada"],
        "anamnesis_flags": ["Refiere posibilidad de embarazo"],
    }
    payload.update(overrides)
    return payload


def test_accepts_a_well_formed_answer(classifier: GeminiClassifier) -> None:
    result = classifier._validate(valid_payload())

    assert result["iga_class"] == "moderado"
    assert result["confidence"] == 0.78
    assert result["gags_score"] == 22
    assert pytest.approx(sum(result["class_probs"].values()), abs=1e-3) == 1.0
    assert len(result["regions"]) == 1


def test_rejects_an_invented_class(classifier: GeminiClassifier) -> None:
    # `iga_class` has a CHECK constraint in the database; anything else fails the INSERT.
    with pytest.raises(GeminiResponseError, match="no es una clase IGA"):
        classifier._validate(valid_payload(iga_class="gravisimo"))


@pytest.mark.parametrize("confidence", [-0.1, 1.4])
def test_rejects_confidence_outside_the_unit_interval(
    classifier: GeminiClassifier, confidence: float
) -> None:
    with pytest.raises(GeminiResponseError, match="fuera de"):
        classifier._validate(valid_payload(confidence=confidence))


def test_renormalises_probabilities_that_are_merely_off_by_rounding(
    classifier: GeminiClassifier,
) -> None:
    probs = {
        "sin_lesiones": 0.02,
        "casi_sin_lesiones": 0.05,
        "leve": 0.10,
        "moderado": 0.78,
        "severo": 0.06,
    }  # sums to 1.01
    result = classifier._validate(valid_payload(class_probs=probs))
    assert pytest.approx(sum(result["class_probs"].values()), abs=1e-3) == 1.0


def test_rejects_probabilities_that_are_wildly_off(classifier: GeminiClassifier) -> None:
    # Renormalising a rounding error is arithmetic; renormalising this would be inventing a
    # distribution the model never expressed.
    probs = dict.fromkeys(["sin_lesiones", "casi_sin_lesiones", "leve", "moderado", "severo"], 1.0)
    with pytest.raises(GeminiResponseError, match="demasiado lejos de 1"):
        classifier._validate(valid_payload(class_probs=probs))


@pytest.mark.parametrize("gags", [-1, 60, 22.5, "22", True])
def test_rejects_gags_outside_its_scale(classifier: GeminiClassifier, gags: object) -> None:
    with pytest.raises(GeminiResponseError):
        classifier._validate(valid_payload(gags_score=gags))


def test_drops_out_of_range_regions_instead_of_clamping(classifier: GeminiClassifier) -> None:
    # A box pointing at the wrong part of the face is worse than no box: the physician would read
    # it as the model's evidence.
    result = classifier._validate(
        valid_payload(
            regions=[
                {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                {"x": 1.4, "y": 0.1, "width": 0.2, "height": 0.2},
                {"x": 0.1, "y": 0.1, "width": 0.0, "height": 0.2},
            ]
        )
    )
    assert len(result["regions"]) == 1


def test_survives_a_missing_regions_field(classifier: GeminiClassifier) -> None:
    payload = valid_payload()
    del payload["regions"]
    result = classifier._validate(payload)
    assert result["regions"] == []


def test_draws_boxes_over_the_image(classifier: GeminiClassifier) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (220, 180, 160)).save(buffer, format="PNG")

    overlay = classifier._draw_regions(buffer.getvalue(), [(0.2, 0.2, 0.4, 0.4)])

    assert overlay is not None
    with Image.open(io.BytesIO(overlay)) as rendered:
        assert rendered.size == (64, 64)
        assert rendered.format == "PNG"


def test_no_regions_means_no_overlay(classifier: GeminiClassifier) -> None:
    # The contract allows a null heatmap, and a clean face has nothing to point at.
    assert classifier._draw_regions(b"", []) is None


def test_requires_an_api_key() -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiClassifier(api_key="", model="gemini-2.5-flash")


def test_reports_itself_as_not_operational() -> None:
    # A general-purpose LLM is not the CNN the design specifies, so it can never gate IIA = 1.
    backend = GeminiClassifier(api_key="k", model="gemini-2.5-flash")
    assert backend.operational is False
    assert backend.kind == "llm_externo"
    assert "gemini" in backend.version


# --- Lectura clínica ---------------------------------------------------------


def test_returns_the_reasoned_reading(classifier: GeminiClassifier) -> None:
    result = classifier._validate(valid_payload())

    assert "pápulas" in result["summary"]
    assert result["lesion_types"] == ("pápulas", "pústulas", "comedones cerrados")
    assert result["affected_zones"][0] == "mejilla derecha"
    assert result["severity_rationale"].startswith("Supera el umbral")
    assert result["anamnesis_flags"] == ("Refiere posibilidad de embarazo",)


def test_rejects_a_grading_with_no_reading(classifier: GeminiClassifier) -> None:
    # A bare class is what the CNN already gives. Asking an LLM and getting no reasoning back means
    # it answered a different question, and a silent empty narrative would look like "no findings".
    with pytest.raises(GeminiResponseError, match="lectura clínica"):
        classifier._validate(valid_payload(summary="   "))


def test_collapses_and_bounds_free_text(classifier: GeminiClassifier) -> None:
    messy = "línea uno\n\n  línea dos   " + "x" * 4000
    result = classifier._validate(valid_payload(summary=messy))

    assert "\n" not in result["summary"]
    assert result["summary"].startswith("línea uno línea dos")
    assert len(result["summary"]) <= 1200


def test_deduplicates_and_caps_lists(classifier: GeminiClassifier) -> None:
    # The model repeats itself across fields; a repeated bullet reads as two separate findings.
    result = classifier._validate(
        valid_payload(lesion_types=["Pápulas", "pápulas", " PÁPULAS ", "", "nódulos"])
    )

    assert result["lesion_types"] == ("Pápulas", "nódulos")


def test_survives_a_model_that_omits_the_optional_lists(classifier: GeminiClassifier) -> None:
    payload = valid_payload()
    for key in ("lesion_types", "differentials", "considerations", "anamnesis_flags"):
        del payload[key]

    result = classifier._validate(payload)

    assert result["lesion_types"] == ()
    assert result["differentials"] == ()


# --- El turno que acompaña a la imagen ---------------------------------------


def test_prompt_without_history_says_so() -> None:
    # Silence about the anamnesis would invite the model to invent flags it was never told.
    prompt = _user_prompt(None)

    assert "No se dispone de anamnesis" in prompt
    assert "anamnesis_flags vacío" in prompt


def test_prompt_quotes_the_history_verbatim() -> None:
    prompt = _user_prompt(
        ClinicalContext(
            age=24,
            sex="femenino",
            phototype="IV",
            answers=(
                AnamnesisItem(question="¿Podría estar embarazada?", answer="Sí"),
                AnamnesisItem(question="¿Hace cuánto aparecieron?", answer="más de 1 año"),
            ),
        )
    )

    assert "24 años" in prompt
    assert "femenino" in prompt
    assert "Fitzpatrick: IV" in prompt
    assert "¿Podría estar embarazada? Sí" in prompt
    # The photograph grades; the history contextualises. Saying so in the turn is what keeps a
    # patient who reports "acné severo" from talking the model into a severe grade.
    assert "la severidad la fija la fotografía" in prompt


def test_an_empty_context_is_treated_as_no_context() -> None:
    assert "No se dispone de anamnesis" in _user_prompt(ClinicalContext())
