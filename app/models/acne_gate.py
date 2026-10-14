"""Acne gate: acne, healthy skin, or another condition? (3-class timm classifier)

Activated with ``ACNE_GATE_ENABLED=true`` and a checkpoint at ``ACNE_GATE_CHECKPOINT``, trained by
``training/gate/train_gate.py``. Without a checkpoint the engine serves exactly as before and says
so (``gate_checked: false`` in every inference).

Why a separate model and not a sixth IGA class: the IGA classifiers were trained and evaluated on
ACNE04 and their F1 is a thesis result. Adding classes means retraining all three and re-running
every comparison. A gate in front leaves them untouched and is measured on its own.

The decision has two steps, and the order is the clinical priority:

1. ``p(acne) ≥ threshold`` → ``acne``. The threshold is chosen on validation for a sensitivity on
   acne of at least 0,97: a real acne case wrongly stopped loses its grade.
2. Otherwise, the likelier of ``sana`` (healthy skin; freckles are a normal variant and belong
   here) and ``otra`` (vitiligo, psoriasis, rosacea…).

The face is cropped with the screener's YuNet detector before anything else. The training sources
differ in what surrounds the face (a white studio wall in CFD, clinic backgrounds in ACNE04), and a
model shown the background will happily learn it instead of the skin.

``face_box`` and ``preprocess_image`` are the ONLY preprocessing for this model: the training
scripts import them, so the picture the gate is trained on and the one it judges in service are
built by the same code.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from app.models.base import GateLabel, GateResult

#: ImageNet normalisation, as for every timm model in the engine (section 8.3).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#: Output columns of the checkpoint, in order. Pinned so a checkpoint saved in another order — or
#: an older binary one — fails loudly instead of mislabelling every verdict.
GATE_CLASSES: tuple[GateLabel, ...] = ("otra", "sana", "acne")
DEFAULT_INPUT_SIZE = 260
#: Margin added around the YuNet box before the square crop, as a share of its larger side.
FACE_MARGIN = 0.15

Box = tuple[int, int, int, int]


def face_box(image: Any, screener: Any | None) -> Box | None:
    """(left, top, right, bottom) of the face in a PIL RGB image, or ``None`` without one.

    ``screener`` is a ``LocalScreener``; ``None`` (no OpenCV) means no crop, the centre is used.
    """
    if screener is None:
        return None
    import numpy as np

    bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
    face = screener.best_face(bgr)
    if face is None:
        return None
    return (face.x, face.y, face.x + face.width, face.y + face.height)


def square_around(image: Any, box: Box | None) -> Any:
    """Square crop centred on ``box`` (plus ``FACE_MARGIN``), or the centred square without one."""
    if box is None:
        side = min(image.width, image.height)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        return image.crop((left, top, left + side, top + side))
    left, top, right, bottom = box
    side = int(max(right - left, bottom - top) * (1 + 2 * FACE_MARGIN))
    side = max(1, min(side, image.width, image.height))
    cx, cy = (left + right) // 2, (top + bottom) // 2
    x0 = min(max(0, cx - side // 2), image.width - side)
    y0 = min(max(0, cy - side // 2), image.height - side)
    return image.crop((x0, y0, x0 + side, y0 + side))


def preprocess(image_bytes: bytes, input_size: int, screener: Any | None = None) -> Any:
    """Decode, find the face, then ``preprocess_image``."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    return preprocess_image(image, input_size, face_box(image, screener))


def preprocess_image(image: Any, input_size: int, box: Box | None = None) -> Any:
    """Square crop on the face, resize, ImageNet normalisation → float32 array HxWx3.

    Takes a PIL RGB image so training can augment it first and still end in this exact function.
    Hue is never touched: the colour of the skin (erythema against brown or white macules) is part
    of what separates acne from its look-alikes.
    """
    import numpy as np
    from PIL import Image

    image = square_around(image, box)
    image = image.resize((input_size, input_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype="float32") / 255.0
    return (array - np.array(IMAGENET_MEAN, dtype="float32")) / np.array(
        IMAGENET_STD, dtype="float32"
    )


def decide(probabilities: dict[str, float], threshold: float) -> GateLabel:
    """Acne first if it clears its threshold; otherwise the likelier of healthy and other."""
    if probabilities.get("acne", 0.0) >= threshold:
        return "acne"
    return "sana" if probabilities.get("sana", 0.0) >= probabilities.get("otra", 0.0) else "otra"


def load_screener() -> Any | None:
    """The YuNet screener used for the face crop, or ``None`` when OpenCV is missing."""
    try:
        from app.models.local_screener import LocalScreener

        return LocalScreener()
    except Exception:  # without it the gate falls back to the centre crop
        return None


class TorchAcneGate:
    """Serves a 3-class timm checkpoint written by ``training/gate/train_gate.py``."""

    def __init__(self, checkpoint_path: str, threshold: float | None = None) -> None:
        try:
            import timm
            import torch
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise RuntimeError("El filtro de acné requiere torch y timm.") from error

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"No existe el checkpoint del filtro de acné en {path}.")

        self._torch = torch
        self.weights_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint: dict[str, Any] = torch.load(path, map_location=self._device)
        classes = tuple(checkpoint.get("classes", ()))
        if classes != GATE_CLASSES:
            raise RuntimeError(
                f"El checkpoint del filtro tiene clases {classes}; se esperaba {GATE_CLASSES}."
            )
        model_name = str(checkpoint.get("model_name", "efficientnet_b0"))
        self.version = str(
            checkpoint.get("version", f"gate_{model_name}_{self.weights_checksum[:8]}")
        )
        self._input_size = int(checkpoint.get("input_size", DEFAULT_INPUT_SIZE))
        # The setting overrides the checkpoint only when given: the checkpoint's own threshold is
        # the one chosen on validation for the target sensitivity.
        self.threshold = float(
            threshold if threshold is not None else checkpoint.get("threshold", 0.5)
        )

        self._model = timm.create_model(model_name, pretrained=False, num_classes=len(GATE_CLASSES))
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval().to(self._device)
        self._screener = load_screener()

    def check(self, image_bytes: bytes, mime_type: str) -> GateResult:
        del mime_type
        import numpy as np

        torch = self._torch
        array = preprocess(image_bytes, self._input_size, self._screener)
        tensor = (torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)).to(
            self._device
        )
        with torch.no_grad():
            output = torch.softmax(self._model(tensor), dim=1)[0]
        probabilities: dict[str, float] = {
            label: round(float(output[index].item()), 4) for index, label in enumerate(GATE_CLASSES)
        }
        return GateResult(
            label=decide(probabilities, self.threshold),
            acne_probability=probabilities["acne"],
            threshold=self.threshold,
            probabilities=probabilities,
        )
