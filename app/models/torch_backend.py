"""Local CNN backend with Grad-CAM interpretability.

Covers: section 8.1 (arquitectura), 8.3 (preprocesamiento) and 8.4 (interpretabilidad).
Reference: Watanabe (2025) — 90,0 % accuracy, F1 macro 0,885, AUC-ROC macro 0,998 with IGA.

Activated with ``MODEL_BACKEND=torch``. Two checkpoint families are served, chosen by
``MODEL_ARCH`` (see ``training/compare.py`` for how the served model is picked):

- ``timm`` — any timm classifier saved by ``training/train.py`` (EfficientNet-B2 of section 8.1,
  ResNet-50 baseline…). The checkpoint carries ``model_name``; without it EfficientNet-B2 is
  assumed.
- ``yolo_cls`` — an Ultralytics YOLOv8 classification checkpoint (``training/train_yolo_cls.py``).
  Ultralytics normalises to [0, 1] with no ImageNet mean/std and its head returns
  ``(probabilities, logits)`` in eval mode; both differences are handled here so the rest of the
  engine sees the same five-class result whatever the architecture.

Imports of torch/timm/captum/ultralytics happen inside the constructor, not at module import
time, so the engine still starts on a machine without the ``[torch]`` extra.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Literal

from app.domain.iga import IGA_CLASSES, IgaClass
from app.models.base import ClassificationResult, ClinicalContext

#: ImageNet normalisation, as section 8.3 prescribes (timm models).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ModelArch = Literal["timm", "yolo_cls"]


class TorchClassifier:
    """Serves a fine-tuned classification checkpoint (timm or YOLOv8-cls)."""

    backend: str = "torch"

    def __init__(
        self,
        checkpoint_path: str,
        seed: int,
        input_size: int,
        enable_gradcam: bool,
        arch: ModelArch = "timm",
    ) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "MODEL_BACKEND=torch requiere el extra [torch]. Instálelo con:\n"
                "  uv pip install -e '.[torch]'\n"
                "En la RTX 5060 (Blackwell, sm_120) use una rueda CUDA 12.8 o superior:\n"
                "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
            ) from error

        self._torch = torch
        self._arch: ModelArch = arch
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

        # Column i of the model's output → position in IGA_CLASSES (None = class the model never
        # saw, e.g. `sin_lesiones` for a YOLO checkpoint trained on ACNE04 folders).
        self._class_positions: list[int | None] = list(range(len(IGA_CLASSES)))
        self._input_size = input_size

        if arch == "yolo_cls":
            self._load_yolo_cls(path)
        else:
            self._load_timm(path)

        self._model.eval().to(self._device)
        self.operational = True
        self.kind = "cnn"

    # --- loading ---------------------------------------------------------------------------

    def _load_timm(self, path: Path) -> None:
        import timm

        checkpoint: dict[str, Any] = self._torch.load(path, map_location=self._device)
        model_name = str(checkpoint.get("model_name", "efficientnet_b2"))
        self.version = str(checkpoint.get("version", f"{model_name}_{self.weights_checksum[:8]}"))
        # The checkpoint knows what it was trained at; the setting is only a fallback for
        # checkpoints that predate the key.
        self._input_size = int(checkpoint.get("input_size", self._input_size))
        self.model_name = model_name

        self._model = timm.create_model(model_name, pretrained=False, num_classes=len(IGA_CLASSES))
        self._model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        self._gradcam_layer = _last_conv_layer(self._model)

    def _load_yolo_cls(self, path: Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "MODEL_ARCH=yolo_cls requiere ultralytics: uv pip install ultralytics"
            ) from error

        wrapper = YOLO(str(path))
        names: dict[int, str] = dict(wrapper.names)
        unknown = sorted(set(names.values()) - set(IGA_CLASSES))
        if unknown:
            raise RuntimeError(f"El checkpoint YOLO tiene clases desconocidas: {unknown}")
        self._class_positions = [IGA_CLASSES.index(names[i]) for i in sorted(names)]

        ckpt = getattr(wrapper, "ckpt", None)
        train_args = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
        self._input_size = int(train_args.get("imgsz", self._input_size))
        self.model_name = "yolov8_cls"
        self.version = f"yolov8_cls_{self.weights_checksum[:8]}"

        self._model = wrapper.model.float()
        # Last block before the `Classify` head: the deepest feature map, where Grad-CAM reads.
        self._gradcam_layer = self._model.model[-2]

    # --- inference -------------------------------------------------------------------------

    def predict(
        self,
        image_bytes: bytes,
        mime_type: str,
        context: ClinicalContext | None = None,
    ) -> ClassificationResult:
        # A CNN sees pixels. The anamnesis is accepted for protocol uniformity and dropped here;
        # `assessment` stays None, and the portal shows the grade without a narrative.
        del context
        torch = self._torch
        tensor = self._preprocess(image_bytes).to(self._device)

        with torch.no_grad():
            logits = self._forward(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]

        # Re-express the model's classes in IGA order; classes it never saw get probability 0.
        full = [0.0] * len(IGA_CLASSES)
        for column, position in enumerate(self._class_positions):
            if position is not None:
                full[position] = float(probabilities[column].item())
        index = max(range(len(full)), key=full.__getitem__)
        predicted: IgaClass = IGA_CLASSES[index]
        class_probs: dict[str, float] = {
            iga_class: round(full[i], 4) for i, iga_class in enumerate(IGA_CLASSES)
        }

        target_column = self._class_positions.index(index)
        heatmap = (
            self._grad_cam(tensor, target_column, image_bytes) if self._enable_gradcam else None
        )

        return ClassificationResult(
            iga_class=predicted,
            confidence=round(full[index], 4),
            class_probs=class_probs,
            heatmap_png=heatmap,
        )

    def _forward(self, tensor: Any) -> Any:
        """Logits, whatever the head returns (Ultralytics gives `(probs, logits)` in eval)."""
        output = self._model(tensor)
        if isinstance(output, (tuple, list)):
            return output[1] if len(output) > 1 else output[0]
        return output

    def _preprocess(self, image_bytes: bytes) -> Any:
        """Face crop, resize to the model's native input and per-architecture normalisation.

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
        if self._arch == "timm":
            array = (array - np.array(IMAGENET_MEAN, dtype="float32")) / np.array(
                IMAGENET_STD, dtype="float32"
            )
        tensor = self._torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).unsqueeze(0)
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
        except Exception:
            return self._centre_square(image)

    @staticmethod
    def _centre_square(image: Any) -> Any:
        side = min(image.width, image.height)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        return image.crop((left, top, left + side, top + side))

    def _grad_cam(self, tensor: Any, target_column: int, original_bytes: bytes) -> bytes | None:
        """Grad-CAM over the last convolutional layer, via Captum (section 8.4)."""
        try:
            import numpy as np
            from captum.attr import LayerAttribution, LayerGradCam
            from PIL import Image
        except ImportError:
            return None

        try:
            grad_cam = LayerGradCam(self._forward, self._gradcam_layer)
            attribution = grad_cam.attribute(tensor, target=target_column, relu_attributions=True)
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
        except Exception:
            return None


def _last_conv_layer(model: Any) -> Any:
    """The deepest convolutional feature map of a timm model, for Grad-CAM.

    EfficientNet exposes it as ``conv_head``, ResNet as ``layer4``; anything else falls back to
    the last ``Conv2d`` registered, which is the last spatial map before pooling in every
    classifier timm ships.
    """
    from torch import nn

    for name in ("conv_head", "layer4", "stages"):
        layer = getattr(model, name, None)
        if layer is not None:
            return layer[-1] if isinstance(layer, nn.Sequential) and name == "stages" else layer
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise RuntimeError("El modelo no tiene capas convolucionales para Grad-CAM.")
    return last
