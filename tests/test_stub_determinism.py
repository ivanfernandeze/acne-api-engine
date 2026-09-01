"""Determinism of the stub classifier.

Covers the testing requirement of section 12.5: "tests deterministas con imagen sintética y
semilla fija". Reproducibility is not a nicety here — RNF-03 requires that any case can be
re-executed identically six months later.
"""

from __future__ import annotations

from app.domain.iga import IGA_CLASSES
from app.models.stub import StubClassifier


def test_same_image_and_seed_yield_identical_prediction(sample_image: bytes) -> None:
    classifier = StubClassifier(seed=42)

    first = classifier.predict(sample_image, "image/png")
    second = classifier.predict(sample_image, "image/png")

    assert first.iga_class == second.iga_class
    assert first.confidence == second.confidence
    assert first.class_probs == second.class_probs


def test_prediction_is_stable_across_instances(sample_image: bytes) -> None:
    """A restarted process must reproduce the prediction exactly."""
    assert (
        StubClassifier(seed=42).predict(sample_image, "image/png").iga_class
        == StubClassifier(seed=42).predict(sample_image, "image/png").iga_class
    )


def test_different_seed_can_change_prediction(sample_image: bytes) -> None:
    """The seed genuinely participates, so it is worth recording alongside each inference."""
    predictions = {
        StubClassifier(seed=seed).predict(sample_image, "image/png").iga_class
        for seed in range(40)
    }
    assert len(predictions) > 1


def test_class_probabilities_form_a_distribution(sample_image: bytes) -> None:
    result = StubClassifier(seed=42).predict(sample_image, "image/png")

    assert set(result.class_probs) == set(IGA_CLASSES)
    assert all(0.0 <= p <= 1.0 for p in result.class_probs.values())
    assert abs(sum(result.class_probs.values()) - 1.0) < 0.01
    # The reported confidence must be the probability of the reported class.
    assert result.class_probs[result.iga_class] == result.confidence
    assert result.confidence == max(result.class_probs.values())


def test_heatmap_is_a_png(sample_image: bytes) -> None:
    result = StubClassifier(seed=42, enable_gradcam=True).predict(sample_image, "image/png")

    assert result.heatmap_png is not None
    assert result.heatmap_png.startswith(b"\x89PNG\r\n\x1a\n")


def test_heatmap_can_be_disabled(sample_image: bytes) -> None:
    result = StubClassifier(seed=42, enable_gradcam=False).predict(sample_image, "image/png")
    assert result.heatmap_png is None


def test_unreadable_image_does_not_break_the_inference() -> None:
    """A corrupt image must still yield a prediction, only without the overlay."""
    result = StubClassifier(seed=42).predict(b"no soy una imagen", "image/png")

    assert result.iga_class in IGA_CLASSES
    assert result.heatmap_png is None


def test_stub_never_claims_to_be_operational() -> None:
    """The gate that stops the thesis from declaring IIA = 1 on a placeholder."""
    assert StubClassifier().operational is False
    assert StubClassifier().weights_checksum is None
