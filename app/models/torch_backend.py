"""EfficientNet-B2 backend with Grad-CAM interpretability.

Covers: section 8.1 (arquitectura), 8.3 (preprocesamiento) and 8.4 (interpretabilidad).
Reference: Watanabe (2025) — 90,0 % accuracy, F1 macro 0,885, AUC-ROC macro 0,998 with IGA.

Activated with ``MODEL_BACKEND=torch``. Imports of torch/timm/captum happen inside the
constructor, not at module import time, so the engine still starts on a machine with no PyTorch —
which is the current state of the study's environment (Python 3.14, no stable wheels).
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from app.domain.iga import IGA_CLASSES, IgaClass
from app.models.base import ClassificationResult

#: ImageNet normalisation, as section 8.3 prescribes.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TorchClassifier:
    """Serves a fine-tuned EfficientNet-B2 checkpoint."""

    backend: str = "torch"

    def __init__(self, checkpoint_path: str, seed: int, input_size: int, enable_gradcam: bool) -> None:
        try:
            import timm
            import torch
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "MODEL_BACKEND=torch requiere el extra [torch]. Instálelo con:\n"
                "  uv pip install -e '.[torch]'\n"
                "En la RTX 5060 (Blackwell, sm_120) use una rueda CUDA 12.8 o superior:\n"
                "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
            ) from error

        self._torch = torch
        self._input_size = input_size
        self._enable_gradcam = enable_gradcam

        # Fixed seed so a rerun six months later reproduces the same prediction (RNF-03).
        torch.manual_seed(seed)
        self._seed = seed

        path = Path(checkpoint_path)
        if not path.is_file():
            raise RuntimeError(
                f"No se encontró el checkpoint del modelo en {path.resolve()}. "
                "Entrene el modelo (training/train.py) o ajuste MODEL_CHECKPOINT_PATH."
            )

        self.weights_checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint: dict[str, Any] = torch.load(path, map_location=self._device)
        self.version = str(checkpoint.get("version", f"efficientnet_b2_{self.weights_checksum[:8]}"))

        self._model = timm.create_model(
            "efficientnet_b2", pretrained=False, num_classes=len(IGA_CLASSES)
        )
        self._model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        self._model.eval().to(self._device)

        self.operational = True

    def predict(self, image_bytes: bytes, mime_type: str) -> ClassificationResult:
        torch = self._torch
        tensor = self._preprocess(image_bytes).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        index = int(torch.argmax(probabilities).item())
        predicted: IgaClass = IGA_CLASSES[index]
        class_probs = {
            iga_class: round(float(probabilities[i].item()), 4)
            for i, iga_class in enumerate(IGA_CLASSES)
        }

        heatmap = self._grad_cam(tensor, index, image_bytes) if self._enable_gradcam else None

        return ClassificationResult(
            iga_class=predicted,
            confidence=round(float(probabilities[index].item()), 4),
            class_probs=class_probs,
            heatmap_png=heatmap,
        )

    def _preprocess(self, image_bytes: bytes) -> Any:
        """Face crop, resize to the model's native input and ImageNet normalisation.

        Hue is deliberately left untouched throughout: shifting it would distort skin tone and
        erythema, the very signals the grading depends on (section 8.3).
        """
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert("RGB")

        image = self._crop_face(image)
        image = image.resize((self._input_size, self._input_size), Image.Resampling.BILINEAR)

        array = np.asarray(image, dtype="float32") / 255.0
        array = (array - np.array(IMAGENET_MEAN, dtype="float32")) / np.array(
            IMAGENET_STD, dtype="float32"
        )
        tensor = self._torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _crop_face(self, image: Any) -> Any:
        """Crop to the face when a detector is available; otherwise use a centred square.

        Falling back rather than failing keeps a hard-to-detect photograph diagnosable: a missed
        crop degrades accuracy, a hard failure loses the case.
        """
        try:
            from facenet_pytorch import MTCNN  # type: ignore[import-not-found]
        except ImportError:
            return self._centre_square(image)

        try:
            detector = MTCNN(keep_all=False, device=self._device)
            boxes, _ = detector.detect(image)
            if boxes is None or len(boxes) == 0:
                return self._centre_square(image)

            x1, y1, x2, y2 = (int(v) for v in boxes[0])
            margin_x = int((x2 - x1) * 0.15)
            margin_y = int((y2 - y1) * 0.15)
            return image.crop(
                (
                    max(0, x1 - margin_x),
                    max(0, y1 - margin_y),
                    min(image.width, x2 + margin_x),
                    min(image.height, y2 + margin_y),
                )
            )
        except Exception:  # noqa: BLE001 - detection is best-effort
            return self._centre_square(image)

    @staticmethod
    def _centre_square(image: Any) -> Any:
        side = min(image.width, image.height)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        return image.crop((left, top, left + side, top + side))

    def _grad_cam(self, tensor: Any, target_index: int, original_bytes: bytes) -> bytes | None:
        """Grad-CAM over the last convolutional layer, via Captum (section 8.4)."""
        try:
            import numpy as np
            from captum.attr import LayerGradCam, LayerAttribution
            from PIL import Image
        except ImportError:
            return None

        try:
            layer = self._model.conv_head
            grad_cam = LayerGradCam(self._model, layer)
            attribution = grad_cam.attribute(tensor, target=target_index, relu_attributions=True)
            upsampled = LayerAttribution.interpolate(
                attribution, (self._input_size, self._input_size), interpolate_mode="bilinear"
            )

            heat = upsampled.squeeze().detach().cpu().numpy()
            span = float(heat.max() - heat.min())
            heat = (heat - heat.min()) / span if span > 0 else np.zeros_like(heat)

            # Blue → red ramp over the original image, the conventional Grad-CAM reading.
            coloured = np.zeros((*heat.shape, 3), dtype="uint8")
            coloured[..., 0] = (heat * 255).astype("uint8")
            coloured[..., 2] = ((1 - heat) * 255).astype("uint8")

            with Image.open(io.BytesIO(original_bytes)) as source:
                base = source.convert("RGB").resize((self._input_size, self._input_size))

            blended = Image.blend(base, Image.fromarray(coloured), alpha=0.45)
            buffer = io.BytesIO()
            blended.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
        except Exception:  # noqa: BLE001 - interpretability must never break a diagnosis
            return None
