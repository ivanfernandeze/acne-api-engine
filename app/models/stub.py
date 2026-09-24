"""Deterministic stub classifier.

Purpose: make the entire clinical and statistical pipeline demonstrable while no CNN has been
trained yet. It is NOT a model — it derives a stable pseudo-prediction from the SHA-256 of the
image bytes — and it says so: ``operational`` is ``False``, which is what keeps
``GET /audit/iia-check`` from ever claiming IIA = 1 on the strength of a placeholder.

Determinism matters twice over: the same image always yields the same class, so a demo is
repeatable and the pytest suite can assert exact equality (RNF-03).
"""

from __future__ import annotations

import hashlib
import io
import math

from PIL import Image, ImageDraw, ImageFilter

from app.domain.iga import IGA_CLASSES, IgaClass
from app.models.base import ClassificationResult, ClinicalContext

STUB_VERSION = "stub-deterministic-v0"


class StubClassifier:
    """Hash-derived classifier with a synthetic Grad-CAM-style overlay."""

    version: str = STUB_VERSION
    backend: str = "stub"
    weights_checksum: str | None = None
    operational: bool = False
    kind: str = "stub"

    def __init__(self, seed: int = 42, enable_gradcam: bool = True) -> None:
        self._seed = seed
        self._enable_gradcam = enable_gradcam

    def predict(
        self,
        image_bytes: bytes,
        mime_type: str,
        context: ClinicalContext | None = None,
    ) -> ClassificationResult:
        # `context` is accepted and ignored on purpose: the protocol has to be uniform so that
        # swapping backends never changes the caller. A hash of the bytes has nothing to say about
        # a clinical history, and pretending otherwise would fabricate a reading.
        del context
        digest = hashlib.sha256(image_bytes + str(self._seed).encode("utf-8")).digest()

        # Byte 0 selects the class; bytes 1-2 shape the confidence. Both are stable for a given
        # image, so repeated calls agree exactly.
        index = digest[0] % len(IGA_CLASSES)
        predicted: IgaClass = IGA_CLASSES[index]
        confidence = 0.62 + (int.from_bytes(digest[1:3], "big") / 65535) * 0.33

        class_probs = self._distribute(predicted, confidence, digest)
        heatmap = self._build_heatmap(image_bytes, digest) if self._enable_gradcam else None

        return ClassificationResult(
            iga_class=predicted,
            confidence=round(confidence, 4),
            class_probs=class_probs,
            heatmap_png=heatmap,
        )

    def _distribute(
        self, predicted: IgaClass, confidence: float, digest: bytes
    ) -> dict[str, float]:
        """Spread the remaining mass over the other classes, weighted by ordinal proximity.

        A plausible probability vector matters for the portal: a flat remainder would make the
        second-best class arbitrary, and the physician reads that ranking.
        """
        predicted_index = IGA_CLASSES.index(predicted)
        weights: list[float] = []
        for i, _ in enumerate(IGA_CLASSES):
            if i == predicted_index:
                weights.append(0.0)
                continue
            distance = abs(i - predicted_index)
            jitter = 0.85 + (digest[(3 + i) % len(digest)] / 255) * 0.3
            weights.append(math.exp(-distance) * jitter)

        total = sum(weights)
        remainder = 1.0 - confidence
        probs: dict[str, float] = {}
        for i, iga_class in enumerate(IGA_CLASSES):
            if i == predicted_index:
                probs[iga_class] = round(confidence, 4)
            else:
                share = 0.0 if total == 0 else weights[i] / total
                probs[iga_class] = round(remainder * share, 4)
        return probs

    def _build_heatmap(self, image_bytes: bytes, digest: bytes) -> bytes | None:
        """Overlay warm blobs on the original image, mimicking a Grad-CAM read-out.

        Explicitly a placeholder: the real attribution arrives with the torch backend via Captum.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                base = source.convert("RGB")
        except Exception:  # noqa: BLE001 - an unreadable image must not fail the inference
            return None

        width, height = base.size
        overlay = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for i in range(3):
            cx = int((digest[8 + i * 3] / 255) * width)
            cy = int((digest[9 + i * 3] / 255) * height)
            radius = int((0.12 + (digest[10 + i * 3] / 255) * 0.14) * min(width, height))
            for step in range(radius, 0, -max(1, radius // 12)):
                intensity = 1.0 - step / radius
                colour = (
                    int(60 + 195 * intensity),
                    int(40 + 120 * max(0.0, 1 - abs(intensity - 0.5) * 2)),
                    int(120 * (1 - intensity)),
                )
                draw.ellipse([cx - step, cy - step, cx + step, cy + step], fill=colour)

        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=max(2, min(width, height) // 60)))
        blended = Image.blend(base, overlay, alpha=0.45)

        buffer = io.BytesIO()
        blended.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
