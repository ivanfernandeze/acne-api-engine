"""Admission screening: what may enter the clinical record.

Two things are pinned here, and both are load-bearing rather than cosmetic:

* a clean, healthy face is ADMISSIBLE — ``sin_lesiones`` is one of the five IGA classes;
* the verdict never carries a clinical judgement, because screening runs at intake in BOTH
  experimental conditions and anything clinical would leak the intervention into the pretest.

None of these tests call the API.
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.models.base import ImageScreener, ScreeningResult
from app.models.gemini_backend import GeminiClassifier, GeminiResponseError

client = TestClient(app)


@pytest.fixture
def classifier() -> GeminiClassifier:
    return GeminiClassifier(api_key="clave-de-prueba", model="gemini-2.5-flash")


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "subject": "rostro_humano",
        "acceptable": True,
        "issues": [],
        "reason": "",
    }
    body.update(overrides)
    return body


def test_accepts_a_facial_photograph(classifier: GeminiClassifier) -> None:
    result = classifier._validate_screening(payload())

    assert result.acceptable is True
    assert result.subject == "rostro_humano"


def test_accepts_a_clean_healthy_face(classifier: GeminiClassifier) -> None:
    # THE test that matters. `sin_lesiones` is a valid IGA class and a resolved case at a follow-up
    # visit looks exactly like this. Rejecting it would throw away the evidence that the treatment
    # worked — and 6 of the 30 study cases are graded `sin_lesiones`.
    result = classifier._validate_screening(
        payload(reason="", issues=[])
    )

    assert result.acceptable is True


@pytest.mark.parametrize(
    "subject", ["persona_sin_rostro", "piel_no_facial", "otro"]
)
def test_rejects_anything_that_is_not_a_face(
    classifier: GeminiClassifier, subject: str
) -> None:
    result = classifier._validate_screening(payload(subject=subject))

    assert result.acceptable is False


def test_the_verdict_is_derived_not_taken_on_trust(classifier: GeminiClassifier) -> None:
    # The model can contradict itself: `acceptable: true` next to a subject that is plainly not a
    # face. Believing the flag would defeat the entire check.
    result = classifier._validate_screening(payload(subject="otro", acceptable=True))

    assert result.acceptable is False


def test_a_doubtful_image_is_admitted(classifier: GeminiClassifier) -> None:
    # A human looks at every photograph anyway. Blocking a real patient's intake over a verdict the
    # model itself could not reach is the worse of the two errors.
    result = classifier._validate_screening(payload(subject="indeterminado", acceptable=False))

    assert result.acceptable is True


def test_quality_problems_warn_but_do_not_block(classifier: GeminiClassifier) -> None:
    result = classifier._validate_screening(
        payload(issues=["enfoque", "iluminacion"], reason="Acérquese y busque mejor luz.")
    )

    assert result.acceptable is True
    assert result.issues == ("enfoque", "iluminacion")
    assert "luz" in result.reason


def test_drops_invented_issue_codes(classifier: GeminiClassifier) -> None:
    result = classifier._validate_screening(payload(issues=["enfoque", "acne_severo", 42]))

    assert result.issues == ("enfoque",)


def test_rejects_an_invented_subject(classifier: GeminiClassifier) -> None:
    with pytest.raises(GeminiResponseError, match="sujeto de imagen"):
        classifier._validate_screening(payload(subject="selfie_bonita"))


def test_the_result_type_carries_nothing_clinical() -> None:
    # Structural, not behavioural: screening runs in the pretest too, so a field able to hold a
    # severity would be a hole in RF-TR-01 waiting for someone to fill it.
    fields = set(ScreeningResult.__dataclass_fields__)

    # `pose` / `yaw_ratio` are head geometry (frontal vs. profile), not a reading of the skin.
    assert fields == {"acceptable", "subject", "issues", "reason", "pose", "yaw_ratio"}
    for forbidden in ("iga", "class", "severity", "lesion", "gags", "confidence"):
        assert not any(forbidden in name for name in fields)


def test_a_backend_that_cannot_screen_says_so_instead_of_guessing() -> None:
    # The suite pins MODEL_BACKEND=stub, and the stub has no `screen`. It must report
    # `checked: false` rather than fabricate a pass that looks like a verification.
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (210, 170, 150)).save(buffer, format="PNG")

    response = client.post(
        "/image/screen",
        json={
            "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "mime_type": "image/png",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["checked"] is False
    assert body["acceptable"] is True
    assert body["subject"] == "indeterminado"


def test_the_stub_is_not_mistaken_for_a_screener() -> None:
    from app.models.stub import StubClassifier

    assert not isinstance(StubClassifier(seed=42), ImageScreener)


def test_rejects_an_empty_image() -> None:
    response = client.post("/image/screen", json={"image_base64": "", "mime_type": "image/png"})

    assert response.status_code == 422


# --- Guardia local, sin llamar a la API --------------------------------------


def flat_image(colour: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (320, 320), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def textured_image() -> bytes:
    image = Image.new("RGB", (320, 320))
    pixels = image.load()
    assert pixels is not None
    for y in range(320):
        for x in range(320):
            pixels[x, y] = ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("colour", [(255, 255, 255), (0, 0, 0), (128, 128, 128)])
def test_a_blank_image_is_rejected_without_calling_the_api(
    classifier: GeminiClassifier, colour: tuple[int, int, int]
) -> None:
    # Decided locally: it costs nothing, spends no quota, and cannot be talked out of by a model
    # that hedges. `screen` would raise on any network call, since the key is fake.
    result = classifier.screen(flat_image(colour), "image/png")

    assert result.acceptable is False
    assert result.subject == "otro"
    assert result.issues == ("no_es_fotografia",)


def test_an_image_with_detail_is_not_treated_as_blank(classifier: GeminiClassifier) -> None:
    # The guard must catch only the obvious. A real photograph reaching it would be rejected
    # without anyone ever looking at it.
    assert classifier._is_blank(textured_image()) is False


def test_undecodable_bytes_are_not_called_blank(classifier: GeminiClassifier) -> None:
    # A failure to measure is not evidence of blankness.
    assert classifier._is_blank(b"esto no es una imagen") is False
