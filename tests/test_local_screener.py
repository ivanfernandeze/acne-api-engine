"""The local screener admits photographs of human faces and nothing else.

No clinical photograph is used. What the detector does on real faces is calibrated offline
(1 406 ACNE04 images, see the module docstring); here the decision rules are exercised with
synthetic images and, where a face is needed, with a stubbed detector.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")

from app.models.base import ImageScreener, ScreeningResult  # noqa: E402
from app.models.local_screener import Face, LocalScreener  # noqa: E402


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _skin_face(saturated: bool = True) -> np.ndarray:
    """A 320×320 canvas with a skin-coloured, textured blob — what a face crop looks like in RGB."""
    rng = np.random.default_rng(0)
    canvas = np.full((320, 320, 3), 240, dtype=np.uint8)
    base = np.array([224, 172, 140] if saturated else [150, 150, 150], dtype=np.int16)
    noise = rng.integers(-18, 18, size=(200, 160, 3), dtype=np.int16)
    canvas[60:260, 80:240] = np.clip(base + noise, 0, 255).astype(np.uint8)
    return canvas


@pytest.fixture
def screener() -> LocalScreener:
    return LocalScreener()


def test_it_is_an_image_screener(screener: LocalScreener) -> None:
    assert isinstance(screener, ImageScreener)
    assert screener.version.startswith("local-yunet")


def test_rejects_undecodable_bytes(screener: LocalScreener) -> None:
    result = screener.screen(b"not an image", "image/png")
    assert result.acceptable is False
    assert result.issues == ("no_es_fotografia",)


def test_rejects_a_flat_image(screener: LocalScreener) -> None:
    result = screener.screen(_png(np.full((256, 256, 3), 200, dtype=np.uint8)), "image/png")
    assert result.acceptable is False
    assert "vacía" in result.reason


@pytest.mark.parametrize("kind", ["noise", "line_drawing", "gradient"])
def test_rejects_images_with_no_human_face(screener: LocalScreener, kind: str) -> None:
    rng = np.random.default_rng(1)
    if kind == "noise":
        array = rng.integers(0, 255, size=(320, 320, 3), dtype=np.uint8)
    elif kind == "gradient":
        ramp = np.linspace(0, 255, 320, dtype=np.uint8)
        array = np.stack([np.tile(ramp, (320, 1))] * 3, axis=-1)
    else:
        array = np.full((320, 320, 3), 255, dtype=np.uint8)
        cv2.ellipse(array, (160, 160), (90, 120), 0, 0, 360, (0, 0, 0), 2)
        cv2.circle(array, (125, 130), 8, (0, 0, 0), 2)
        cv2.circle(array, (195, 130), 8, (0, 0, 0), 2)
        cv2.line(array, (160, 150), (160, 190), (0, 0, 0), 2)
        cv2.ellipse(array, (160, 215), (30, 12), 0, 0, 180, (0, 0, 0), 2)
    result = screener.screen(_png(array), "image/png")
    assert result.acceptable is False
    assert result.subject == "otro"
    assert "rostro" in result.reason


def _with_fake_face(screener: LocalScreener, score: float) -> None:
    def fake(image):  # type: ignore[no-untyped-def]
        h, w = image.shape[:2]
        scale = 640 / max(h, w)
        analysed = cv2.resize(image, (int(w * scale), int(h * scale)))
        face = Face(
            x=int(80 * scale),
            y=int(60 * scale),
            width=int(160 * scale),
            height=int(200 * scale),
            score=score,
        )
        return [face], analysed

    screener._detect = fake  # type: ignore[method-assign]


def test_accepts_a_detected_face_with_real_skin_colour(screener: LocalScreener) -> None:
    _with_fake_face(screener, score=0.9)
    # PIL writes RGB; the screener decodes to BGR, so the fixture is built as RGB.
    result = screener.screen(_png(_skin_face(saturated=True)), "image/png")
    assert result == ScreeningResult(acceptable=True, subject="rostro_humano", issues=(), reason="")


def test_rejects_a_detected_face_without_skin_colour_as_an_illustration(
    screener: LocalScreener,
) -> None:
    _with_fake_face(screener, score=0.9)
    result = screener.screen(_png(_skin_face(saturated=False)), "image/png")
    assert result.acceptable is False
    assert result.issues == ("no_es_fotografia",)
    assert "dibujo" in result.reason


def test_a_weak_detection_does_not_count(screener: LocalScreener) -> None:
    _with_fake_face(screener, score=0.2)
    result = screener.screen(_png(_skin_face()), "image/png")
    assert result.acceptable is False
    assert "rostro" in result.reason


def test_quality_remarks_are_warnings_not_rejections(screener: LocalScreener) -> None:
    def tiny_and_double(image):  # type: ignore[no-untyped-def]
        h, w = image.shape[:2]
        scale = 640 / max(h, w)
        analysed = cv2.resize(image, (int(w * scale), int(h * scale)))
        small = Face(
            x=int(120 * scale),
            y=int(120 * scale),
            width=int(40 * scale),
            height=int(50 * scale),
            score=0.8,
        )
        other = Face(
            x=int(200 * scale),
            y=int(120 * scale),
            width=int(40 * scale),
            height=int(50 * scale),
            score=0.7,
        )
        return [small, other], analysed

    screener._detect = tiny_and_double  # type: ignore[method-assign]
    result = screener.screen(_png(_skin_face()), "image/png")
    assert result.acceptable is True
    assert "encuadre" in result.issues


def test_the_result_carries_nothing_clinical(screener: LocalScreener) -> None:
    _with_fake_face(screener, score=0.9)
    result = screener.screen(_png(_skin_face()), "image/png")
    for field in vars(result):
        assert not any(word in field for word in ("iga", "class", "severity", "lesion", "gags"))
