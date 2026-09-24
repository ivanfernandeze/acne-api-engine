"""Several trained classifiers, run together on every image, combined into one recommendation.

Covers: section 8.1 (revision 17/09/2026) — the models compared in ``training/compare.py`` are
served side by side instead of one at a time. Activated with ``MODEL_BACKEND=torch`` and
``MODEL_ARCH=ensemble``; ``ENSEMBLE_MEMBERS`` lists them as ``tag:arch:path:weight``.

**How the recommendation is formed** (``ENSEMBLE_RULE``):

- ``mas_confiable`` (default since 18/09/2026, decision of the research team): the member with
  the highest confidence gives the class, the probability vector and the lead Grad-CAM; a tie goes
  to the member with the higher weight (F1). The suggestion is always something one model actually
  read, which is what the specialist asked to see. The known cost: a model's confidence on one
  image is not its accuracy on that image — the per-member table and the agreement level are shown
  precisely so an over-confident wrong vote can be spotted.
- ``voto_ponderado``: the five probabilities averaged, weighted by each member's F1 macro on the
  frozen test split; arg-max of the mean. Statistically sounder (it is the classic soft vote) but
  it can name a class no member ranked first on an ordinal scale, which read as wrong in the
  clinic.

The individual readings are returned whole either way, with their own Grad-CAM.

**What it does NOT do.** It never drops a member silently: a checkpoint that fails to load fails
the whole backend, and the registry falls back to the stub with ``operational: false``. Serving
"the ensemble" with two of three models would be a different model than the one documented.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Literal

from app.domain.iga import IGA_CLASSES, IgaClass
from app.models.base import Agreement, ClassificationResult, ClinicalContext, MemberPrediction
from app.models.torch_backend import ModelArch, TorchClassifier

EnsembleRule = Literal["mas_confiable", "voto_ponderado"]


@dataclass(frozen=True)
class MemberSpec:
    tag: str
    arch: ModelArch
    checkpoint_path: str
    weight: float


def parse_members(spec: str) -> list[MemberSpec]:
    """``tag:arch:path:weight, ...`` → specs. A malformed entry is an error, never skipped."""
    members: list[MemberSpec] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        parts = entry.split(":")
        # Windows paths carry a drive colon (`C:/...`); take tag and arch from the front, the
        # weight from the back, and whatever is left in the middle is the path.
        if len(parts) < 4:
            raise ValueError(f"ENSEMBLE_MEMBERS: entrada inválida '{entry}' (tag:arch:path:peso)")
        tag, arch, weight = parts[0], parts[1], parts[-1]
        path = ":".join(parts[2:-1])
        if arch not in ("timm", "yolo_cls"):
            raise ValueError(f"ENSEMBLE_MEMBERS: arquitectura desconocida '{arch}' en '{entry}'")
        try:
            parsed_weight = float(weight)
        except ValueError as error:
            raise ValueError(f"ENSEMBLE_MEMBERS: peso inválido '{weight}' en '{entry}'") from error
        if parsed_weight <= 0:
            raise ValueError(f"ENSEMBLE_MEMBERS: el peso debe ser > 0 en '{entry}'")
        members.append(MemberSpec(tag, arch, path, parsed_weight))  # type: ignore[arg-type]
    if len(members) < 2:
        raise ValueError("ENSEMBLE_MEMBERS necesita al menos dos modelos.")
    if len({m.tag for m in members}) != len(members):
        raise ValueError("ENSEMBLE_MEMBERS: las etiquetas deben ser únicas.")
    return members


class EnsembleClassifier:
    """Weighted soft-vote over several ``TorchClassifier`` members."""

    backend: str = "torch"

    def __init__(
        self,
        members: list[MemberSpec],
        seed: int,
        input_size: int,
        enable_gradcam: bool,
        rule: EnsembleRule = "mas_confiable",
    ) -> None:
        self._rule: EnsembleRule = rule
        self._members: list[tuple[MemberSpec, TorchClassifier]] = []
        for spec in members:
            classifier = TorchClassifier(
                checkpoint_path=spec.checkpoint_path,
                seed=seed,
                input_size=input_size,
                enable_gradcam=enable_gradcam,
                arch=spec.arch,
            )
            self._members.append((spec, classifier))

        # One checksum for the whole ensemble, derived from the members' in a fixed order, so
        # /audit/iia-check still publishes a single value that changes if any weight file changes.
        digest = hashlib.sha256()
        for spec, classifier in self._members:
            digest.update(f"{spec.tag}:{classifier.weights_checksum}:{spec.weight}".encode())
        self.weights_checksum = digest.hexdigest()
        tag = "max" if rule == "mas_confiable" else "avg"
        self.version = f"ensemble-{tag}[" + "+".join(c.version for _, c in self._members) + "]"
        self.model_name = "ensemble"
        self.operational = True
        self.kind = "cnn"
        self._seed = seed

    @property
    def members(self) -> list[tuple[MemberSpec, TorchClassifier]]:
        return list(self._members)

    def predict(
        self,
        image_bytes: bytes,
        mime_type: str,
        context: ClinicalContext | None = None,
    ) -> ClassificationResult:
        readings: list[MemberPrediction] = []
        combined = [0.0] * len(IGA_CLASSES)
        total_weight = 0.0

        for spec, classifier in self._members:
            started = time.perf_counter()
            result = classifier.predict(image_bytes, mime_type, context)
            latency_ms = round((time.perf_counter() - started) * 1000)
            readings.append(
                MemberPrediction(
                    tag=spec.tag,
                    model_name=classifier.model_name,
                    model_version=classifier.version,
                    weights_checksum=classifier.weights_checksum,
                    weight=spec.weight,
                    iga_class=result.iga_class,
                    confidence=result.confidence,
                    class_probs=result.class_probs,
                    heatmap_png=result.heatmap_png,
                    latency_ms=latency_ms,
                )
            )
            for i, iga_class in enumerate(IGA_CLASSES):
                combined[i] += spec.weight * result.class_probs.get(iga_class, 0.0)
            total_weight += spec.weight

        if self._rule == "mas_confiable":
            # The surest member speaks for the ensemble; ties go to the better model on the test.
            surest = max(readings, key=lambda r: (r.confidence, r.weight))
            return ClassificationResult(
                iga_class=surest.iga_class,
                confidence=surest.confidence,
                class_probs=dict(surest.class_probs),
                heatmap_png=surest.heatmap_png,
                members=tuple(readings),
                agreement=agreement_of([r.iga_class for r in readings]),
            )

        combined = [value / total_weight for value in combined]
        index = max(range(len(combined)), key=combined.__getitem__)
        recommended: IgaClass = IGA_CLASSES[index]
        class_probs = {c: round(combined[i], 4) for i, c in enumerate(IGA_CLASSES)}

        # The lead Grad-CAM is the heaviest member's; every member's map travels in `members`.
        lead = max(readings, key=lambda r: r.weight)

        return ClassificationResult(
            iga_class=recommended,
            confidence=round(combined[index], 4),
            class_probs=class_probs,
            heatmap_png=lead.heatmap_png,
            members=tuple(readings),
            agreement=agreement_of([r.iga_class for r in readings]),
        )


def agreement_of(votes: list[IgaClass]) -> Agreement:
    """`unanime` when every member says the same, `mayoria` when more than half do."""
    if not votes:
        return "discrepancia"
    top = max(votes.count(v) for v in set(votes))
    if top == len(votes):
        return "unanime"
    if top * 2 > len(votes):
        return "mayoria"
    return "discrepancia"
