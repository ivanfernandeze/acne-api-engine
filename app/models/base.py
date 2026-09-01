"""Classifier interface shared by the stub and the real CNN backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.domain.iga import IgaClass


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of one inference, independent of which backend produced it."""

    iga_class: IgaClass
    confidence: float
    class_probs: dict[str, float]
    #: Grad-CAM overlay as PNG bytes; ``None`` when interpretability is disabled.
    heatmap_png: bytes | None


class Classifier(Protocol):
    """Contract every backend must satisfy.

    Keeping the stub and the CNN behind one protocol is what lets the whole clinical pipeline —
    queue, Grad-CAM display, TPD, confusion matrix, HE2 contrast — be built and verified before a
    single epoch of training has run, and then switched over without touching a caller.
    """

    #: Reported in every inference and in the IIA checklist.
    version: str
    backend: str
    #: SHA-256 of the weights file; ``None`` when there are no weights.
    weights_checksum: str | None
    #: True only when a genuinely trained model is serving, which is what gates IIA = 1.
    operational: bool

    def predict(self, image_bytes: bytes, mime_type: str) -> ClassificationResult:
        """Classify one image into the five IGA classes."""
        ...
