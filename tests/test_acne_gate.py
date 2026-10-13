"""The acne gate decides what reaches the IGA models: acne, healthy skin, or another condition.

The real gate is a trained checkpoint; here it is replaced by fakes, so what is pinned is the
contract of ``/infer/single`` around it: healthy skin is answered as ``sin_lesiones`` without
running the ensemble, another condition carries no grade at all, a failing gate fails open, and no
gate is reported as ``gate_checked: false``.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.models import registry
from app.models.acne_gate import GATE_CLASSES, decide, preprocess, square_around
from app.models.base import AcneGate, GateLabel, GateResult

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


class BrokenGate(FakeGate):
    def check(self, image_bytes: bytes, mime_type: str) -> GateResult:
        raise RuntimeError("CUDA out of memory")


@pytest.fixture
def use_gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    def install(gate: object | None) -> None:
        monkeypatch.setattr("app.routers.infer.get_acne_gate", lambda: gate)
        monkeypatch.setattr("app.routers.audit.get_acne_gate", lambda: gate)

    yield install


def _infer(image: bytes) -> dict[str, object]:
    response = client.post(
        "/infer/single",
        json={"image_base64": base64.b64encode(image).decode("ascii"), "mime_type": "image/png"},
    )
    assert response.status_code == 200
    return response.json()


def test_the_fake_satisfies_the_protocol() -> None:
    assert isinstance(FakeGate(0.9), AcneGate)


def test_another_condition_is_not_graded(use_gate, sample_image: bytes) -> None:  # type: ignore[no-untyped-def]
    use_gate(FakeGate(acne=0.08, sana=0.10))

    body = _infer(sample_image)

    assert body["verdict"] == "no_acne"
    assert body["gate_checked"] is True
    assert body["acne_probability"] == pytest.approx(0.08)
    assert body["gate_threshold"] == pytest.approx(0.5)
    assert body["gate_version"] == "gate_fake_v1"
    assert body["gate_probabilities"] == pytest.approx({"acne": 0.08, "sana": 0.10, "otra": 0.82})
    # Not graded at all: no class, no GAGS, no heatmap, no member readings.
    assert body["prediction"] is None
    assert body["gags_secondary"] is None
    assert body["heatmap_png_base64"] is None
    assert body["members"] == []


def test_healthy_skin_is_answered_as_sin_lesiones_by_the_gate(
    use_gate, sample_image: bytes
) -> None:  # type: ignore[no-untyped-def]
    use_gate(FakeGate(acne=0.05, sana=0.90))

    body = _infer(sample_image)

    assert body["verdict"] == "piel_sana"
    # The class ACNE04 never let the IGA models learn, with the GATE's probability.
    assert body["prediction"] == {
        "iga_class": "sin_lesiones",
        "confidence": pytest.approx(0.90),
        "class_probs": {"sin_lesiones": pytest.approx(0.90)},
    }
    assert body["gags_secondary"] == {"grade": "sin_acne", "score": 0}
    # The ensemble did not run: nothing of its own to show.
    assert body["members"] == []
    assert body["heatmap_png_base64"] is None


def test_an_acne_image_is_graded_as_before(use_gate, sample_image: bytes) -> None:  # type: ignore[no-untyped-def]
    use_gate(FakeGate(acne=0.93, sana=0.05))

    body = _infer(sample_image)

    assert body["verdict"] == "acne"
    assert body["gate_checked"] is True
    assert body["prediction"]["iga_class"] is not None  # type: ignore[index]


def test_acne_wins_once_it_clears_its_threshold_even_if_not_the_top_class() -> None:
    # Sensitivity first: a real acne case wrongly stopped loses its grade.
    assert decide({"acne": 0.35, "sana": 0.60, "otra": 0.05}, threshold=0.30) == "acne"
    assert decide({"acne": 0.25, "sana": 0.60, "otra": 0.15}, threshold=0.30) == "sana"
    assert decide({"acne": 0.25, "sana": 0.15, "otra": 0.60}, threshold=0.30) == "otra"


def test_a_failing_gate_fails_open(use_gate, sample_image: bytes) -> None:  # type: ignore[no-untyped-def]
    use_gate(BrokenGate(acne=0.0))

    body = _infer(sample_image)

    assert body["verdict"] == "acne"
    assert body["gate_checked"] is False
    assert body["prediction"] is not None


def test_without_a_gate_nothing_is_claimed(use_gate, sample_image: bytes) -> None:  # type: ignore[no-untyped-def]
    use_gate(None)

    body = _infer(sample_image)

    assert body["verdict"] == "acne"
    assert body["gate_checked"] is False
    assert body["acne_probability"] is None
    assert body["gate_probabilities"] is None


def test_the_iia_check_says_when_there_is_no_gate(use_gate) -> None:  # type: ignore[no-untyped-def]
    use_gate(None)
    body = client.get("/audit/iia-check").json()
    assert body["acne_gate_version"] is None
    assert any("filtro de acné" in warning for warning in body["warnings"])

    use_gate(FakeGate(0.9))
    body = client.get("/audit/iia-check").json()
    assert body["acne_gate_version"] == "gate_fake_v1"
    assert not any("filtro de acné" in warning for warning in body["warnings"])


def test_registry_returns_no_gate_without_a_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("ACNE_GATE_ENABLED", "true")
    monkeypatch.setenv("ACNE_GATE_CHECKPOINT", str(tmp_path / "missing.pt"))
    from app.config import get_settings

    get_settings.cache_clear()
    registry.get_acne_gate.cache_clear()
    try:
        assert registry.get_acne_gate() is None
    finally:
        get_settings.cache_clear()
        registry.get_acne_gate.cache_clear()


def test_the_checkpoint_class_order_is_pinned() -> None:
    # A checkpoint saved in another order (or an old binary one) would mislabel every verdict.
    assert GATE_CLASSES == ("otra", "sana", "acne")


def test_preprocess_keeps_the_shape(sample_image: bytes) -> None:
    array = preprocess(sample_image, 32)
    assert array.shape == (32, 32, 3)


def test_without_a_face_the_crop_is_the_centred_square() -> None:
    image = Image.new("RGB", (300, 200))
    assert square_around(image, None).size == (200, 200)


def test_the_face_crop_stays_inside_the_image() -> None:
    image = Image.new("RGB", (300, 200))
    # A face at the right edge: the square is shifted in, never padded with nothing.
    crop = square_around(image, (250, 50, 300, 120))
    assert crop.size[0] == crop.size[1]
    assert crop.size[0] <= 200
