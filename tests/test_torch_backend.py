"""The torch backend serves any timm checkpoint, and a YOLOv8-cls one, through one contract.

No trained weights are involved: a randomly initialised `resnet18` saved the way
`training/train.py` saves is enough to prove loading, the five-class output, the IGA class
remapping and the Grad-CAM path. The YOLO case is skipped where ultralytics is not installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from app.domain.iga import IGA_CLASSES  # noqa: E402
from app.models.torch_backend import TorchClassifier, _last_conv_layer  # noqa: E402


@pytest.fixture
def timm_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False, num_classes=len(IGA_CLASSES))
    path = tmp_path / "resnet18_iga.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": "resnet18",
            "version": "resnet18_test",
            "input_size": 64,
            "classes": list(IGA_CLASSES),
        },
        path,
    )
    return path


def test_timm_checkpoint_is_served_with_the_five_iga_classes(
    timm_checkpoint: Path, sample_image: bytes
) -> None:
    classifier = TorchClassifier(
        checkpoint_path=str(timm_checkpoint), seed=42, input_size=999, enable_gradcam=True
    )

    assert classifier.kind == "cnn"
    assert classifier.operational is True
    assert classifier.version == "resnet18_test"
    assert classifier.model_name == "resnet18"
    # The checkpoint's own input size wins over the setting.
    assert classifier._input_size == 64

    result = classifier.predict(sample_image, "image/png")
    assert result.iga_class in IGA_CLASSES
    assert set(result.class_probs) == set(IGA_CLASSES)
    assert abs(sum(result.class_probs.values()) - 1.0) < 0.01
    assert result.class_probs[result.iga_class] == result.confidence
    assert result.assessment is None
    # Grad-CAM found `layer4` on a ResNet and produced a PNG.
    assert result.heatmap_png is not None and result.heatmap_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_prediction_is_deterministic(timm_checkpoint: Path, sample_image: bytes) -> None:
    classifier = TorchClassifier(
        checkpoint_path=str(timm_checkpoint), seed=42, input_size=64, enable_gradcam=False
    )
    first = classifier.predict(sample_image, "image/png")
    second = classifier.predict(sample_image, "image/png")
    assert first.class_probs == second.class_probs


def test_missing_checkpoint_is_a_loud_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="checkpoint"):
        TorchClassifier(
            checkpoint_path=str(tmp_path / "nope.pt"), seed=42, input_size=64, enable_gradcam=False
        )


def test_last_conv_layer_resolves_for_common_families() -> None:
    from torch import nn

    assert isinstance(_last_conv_layer(timm.create_model("resnet18", pretrained=False)), nn.Module)
    efficientnet = timm.create_model("efficientnet_b0", pretrained=False)
    assert _last_conv_layer(efficientnet) is efficientnet.conv_head


def test_yolo_cls_checkpoint_maps_folder_classes_onto_iga(
    tmp_path: Path, sample_image: bytes
) -> None:
    ultralytics = pytest.importorskip("ultralytics")

    # A YOLOv8-cls trained on ACNE04 folders knows four classes, numbered alphabetically by
    # Ultralytics; the backend has to put them back in IGA order and give `sin_lesiones` 0.
    model = ultralytics.YOLO("yolov8n-cls.yaml")
    names = {0: "casi_sin_lesiones", 1: "leve", 2: "moderado", 3: "severo"}
    model.model.names = names
    model.model.nc = 4
    # Replace the head so the output width matches the four names.
    from ultralytics.nn.modules.head import Classify

    head = model.model.model[-1]
    model.model.model[-1] = Classify(head.conv.conv.in_channels, 4)
    model.model.model[-1].f = head.f
    model.model.model[-1].i = head.i
    path = tmp_path / "yolo_cls.pt"
    torch.save({"model": model.model, "train_args": {"imgsz": 64}}, path)

    classifier = TorchClassifier(
        checkpoint_path=str(path), seed=42, input_size=999, enable_gradcam=False, arch="yolo_cls"
    )
    assert classifier._input_size == 64
    assert classifier.model_name == "yolov8_cls"

    result = classifier.predict(sample_image, "image/png")
    assert result.class_probs["sin_lesiones"] == 0.0
    assert result.iga_class != "sin_lesiones"
    assert abs(sum(result.class_probs.values()) - 1.0) < 0.01
