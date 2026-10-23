"""The second opinion before grading: advisory, parallel, and never a reason to lose a grade.

No test here calls Google: the checker is replaced by fakes, and the Gemini adapter is exercised
through `parse_opinion` and `prepare_for_upload`, which are pure.
"""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.models.acne_gate import decide
from app.models.base import GateLabel, GateResult, SecondOpinion, SkinConditionChecker
from app.models.condition_check import MAX_SIDE, parse_opinion, prepare_for_upload
from app.models.gemini_backend import GeminiResponseError

client = TestClient(app)


class FakeGate:
    version = "gate_fake_v1"
    weights_checksum = "0" * 64

    def __init__(self, acne: float, sana: float = 0.0, threshold: float = 0.5) -> None:
        self.probabilities = {"acne": acne, "sana": sana, "otra": round(1 - acne - sana, 4)}
        self.threshold = threshold

    def check(self, image_bytes: bytes, mime_type: str) -> GateResult:
        label: GateLabel = decide(self.probabilities, self.threshold)
        return GateResult(
            label=label,
            acne_probability=self.probabilities["acne"],
            threshold=self.threshold,
            probabilities=self.probabilities,
        )


class FakeChecker:
    version = "gemini/fake/c1"

    def __init__(self, opinion: SecondOpinion) -> None:
        self.opinion = opinion
        self.calls = 0

    def check(self, image_bytes: bytes, mime_type: str) -> SecondOpinion:
        self.calls += 1
        return self.opinion


class BrokenChecker(FakeChecker):
    def check(self, image_bytes: bytes, mime_type: str) -> SecondOpinion:
        raise GeminiResponseError("La API de Gemini respondió 503")


VITILIGO = SecondOpinion(
    label="otra_condicion",
    conditions=("vitíligo",),
    explanation="Máculas acrómicas de bordes netos en mejilla y ala nasal, sin pápulas.",
)


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    def _install(gate: object | None, checker: object | None) -> None:
        monkeypatch.setattr("app.routers.infer.get_acne_gate", lambda: gate)
        monkeypatch.setattr("app.routers.infer.get_second_opinion", lambda: checker)

    yield _install


def _infer(image: bytes) -> dict[str, object]:
    response = client.post(
        "/infer/single",
        json={"image_base64": base64.b64encode(image).decode("ascii"), "mime_type": "image/png"},
    )
    assert response.status_code == 200
    return response.json()


def test_the_fake_satisfies_the_protocol() -> None:
    assert isinstance(FakeChecker(VITILIGO), SkinConditionChecker)


def test_another_condition_stops_the_grading(install, sample_image: bytes) -> None:  # type: ignore[no-untyped-def]
    # The real case: the local gate let a vitiligo close-up through as acne (0.995). The second
    # opinion saw vitiligo, so the image is not graded — the same rule as the gate's own no_acne —
    # and the record says who decided.
    install(FakeGate(acne=0.995), FakeChecker(VITILIGO))
    body = _infer(sample_image)
    assert body["verdict"] == "no_acne"
    assert body["verdict_source"] == "segunda_opinion"
    assert body["prediction"] is None
    assert body["gags_secondary"] is None
    assert body["acne_probability"] == 0.995
    opinion = body["second_opinion"]
    assert isinstance(opinion, dict)
    assert opinion["label"] == "otra_condicion"
    assert opinion["conditions"] == ["vitíligo"]
    assert opinion["version"] == "gemini/fake/c1"


def test_advisory_mode_grades_and_only_warns(  # type: ignore[no-untyped-def]
    install, sample_image: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "second_opinion_stops_grading", False)
    install(FakeGate(acne=0.995), FakeChecker(VITILIGO))
    body = _infer(sample_image)
    assert body["verdict"] == "acne"
    assert body["verdict_source"] == "filtro"
    assert body["prediction"] is not None
    assert body["second_opinion"]["conditions"] == ["vitíligo"]  # type: ignore[index]


def test_healthy_skin_from_the_second_opinion_is_still_graded(install, sample_image: bytes) -> None:  # type: ignore[no-untyped-def]
    # Grade 0 is on the IGA scale: grading healthy skin is not a contradiction, only a warning.
    healthy = SecondOpinion(label="piel_sana", conditions=(), explanation="Piel sin lesiones.")
    install(FakeGate(acne=0.9), FakeChecker(healthy))
    body = _infer(sample_image)
    assert body["verdict"] == "acne"
    assert body["prediction"] is not None
    assert body["second_opinion"]["label"] == "piel_sana"  # type: ignore[index]


def test_images_the_gate_already_stopped_are_not_sent(install, sample_image: bytes) -> None:  # type: ignore[no-untyped-def]
    # Every call is a photograph leaving the machine: only images about to be graded go out.
    checker = FakeChecker(VITILIGO)
    install(FakeGate(acne=0.01, sana=0.9), checker)
    body = _infer(sample_image)
    assert body["verdict"] == "piel_sana"
    assert body["second_opinion"] is None
    assert checker.calls == 0


def test_a_failing_second_opinion_never_costs_the_grade(install, sample_image: bytes) -> None:  # type: ignore[no-untyped-def]
    install(FakeGate(acne=0.9), BrokenChecker(VITILIGO))
    body = _infer(sample_image)
    assert body["verdict"] == "acne"
    assert body["prediction"] is not None
    assert body["second_opinion"] is None


def test_without_a_second_opinion_nothing_is_claimed(install, sample_image: bytes) -> None:  # type: ignore[no-untyped-def]
    install(FakeGate(acne=0.9), None)
    assert _infer(sample_image)["second_opinion"] is None


def test_parse_keeps_conditions_and_drops_freckles() -> None:
    opinion = parse_opinion(
        {
            "label": "acne",
            "conditions": ["Vitíligo", "pecas", "vitíligo", "  melasma  ", 7],
            "explanation": "  Pápulas   inflamatorias en mejilla. ",
        }
    )
    assert opinion.label == "acne"
    assert opinion.conditions == ("Vitíligo", "melasma")
    assert opinion.explanation == "Pápulas inflamatorias en mejilla."


def test_parse_refuses_an_unknown_label() -> None:
    with pytest.raises(GeminiResponseError):
        parse_opinion({"label": "severo", "conditions": [], "explanation": ""})


def test_the_copy_sent_is_small_jpeg_without_metadata() -> None:
    image = Image.new("RGB", (3000, 2000), (180, 140, 120))
    exif = Image.Exif()
    exif[0x010F] = "Fabricante del teléfono"  # Make
    exif[0x0132] = "2026:10:19 17:30:00"  # DateTime
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    payload, mime = prepare_for_upload(buffer.getvalue())

    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(payload)) as sent:
        assert max(sent.size) == MAX_SIDE
        assert len(sent.getexif()) == 0
