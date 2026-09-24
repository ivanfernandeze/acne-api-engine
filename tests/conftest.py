"""Shared fixtures.

Images are synthesised in-process: the test suite must never depend on clinical photographs,
and a fixed seed keeps every generated image byte-identical across runs (RNF-03).
"""

from __future__ import annotations

import io
import os

# The suite pins the stub backend BEFORE anything imports `app`, because `Settings` reads `.env`
# and `get_classifier` caches its answer for the life of the process. Without this, a developer
# with `MODEL_BACKEND=gemini` in their `.env` would send every test image to a paid external API
# and make the contract tests depend on the network — and `test_infer_single_is_deterministic`
# would be asserting the determinism of a remote LLM, which nothing guarantees.
# `GeminiClassifier` is exercised by `test_gemini_backend.py`, which calls no API at all.
os.environ["MODEL_BACKEND"] = "stub"
# The local screener is unit-tested on its own; the route tests want the "nothing screens" path.
os.environ["SCREENING_BACKEND"] = "none"

import pytest
from PIL import Image


def _make_image(seed: int, size: int = 64) -> bytes:
    """Deterministic RGB test image."""
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    assert pixels is not None
    for y in range(size):
        for x in range(size):
            pixels[x, y] = (
                (x * 3 + seed) % 256,
                (y * 5 + seed * 2) % 256,
                (x * y + seed * 7) % 256,
            )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_image() -> bytes:
    return _make_image(seed=1)


@pytest.fixture
def other_image() -> bytes:
    return _make_image(seed=2)
