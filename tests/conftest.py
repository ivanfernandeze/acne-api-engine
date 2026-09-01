"""Shared fixtures.

Images are synthesised in-process: the test suite must never depend on clinical photographs,
and a fixed seed keeps every generated image byte-identical across runs (RNF-03).
"""

from __future__ import annotations

import io

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
