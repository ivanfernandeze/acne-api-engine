"""Application settings for the acne AI engine.

Covers: RNF-03 (reproducibility) — the seed, the model backend and the dataset version all live
here and travel with every inference response.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ModelBackend = Literal["stub", "torch", "gemini"]
TrackingDriver = Literal["file", "mlflow"]


class Settings(BaseSettings):
    """Environment-driven configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    ai_host: str = "0.0.0.0"
    ai_port: int = 8000
    log_level: str = "info"

    #: ``stub`` keeps the whole clinical pipeline demonstrable before a model exists;
    #: ``torch`` serves the real EfficientNet-B2; ``gemini`` calls a multimodal LLM over the
    #: network. Only ``torch`` can ever report ``operational = True``. See ``app.models.registry``.
    model_backend: ModelBackend = "stub"
    #: Checkpoint family served by the torch backend: any timm classifier saved by
    #: training/train.py, or an Ultralytics YOLOv8-cls checkpoint (training/train_yolo_cls.py).
    #: Which one is served follows training/compare.py, the model-selection rule of the thesis.
    model_arch: Literal["timm", "yolo_cls", "ensemble"] = "timm"
    model_checkpoint_path: str = "./checkpoints/efficientnet_b2_iga.pt"
    #: With MODEL_ARCH=ensemble: the members, comma-separated as `tag:arch:path:weight`. The
    #: weight is the model's F1 macro on the frozen test split (training/compare.py), which is
    #: what the soft vote uses. Every member runs on every image; `prediction` is the combination.
    #: How the members' readings become one: `mas_confiable` takes the member with the highest
    #: confidence (its class, its probabilities, its Grad-CAM; ties go to the higher weight);
    #: `voto_ponderado` averages the five probabilities weighted by each member's F1.
    ensemble_rule: Literal["mas_confiable", "voto_ponderado"] = "mas_confiable"
    ensemble_members: str = (
        "yolov8s-cls:yolo_cls:./checkpoints/yolov8s_cls_iga.pt:0.676,"
        "efficientnet_b2:timm:./checkpoints/efficientnet_b2_iga.pt:0.633,"
        "resnet50:timm:./checkpoints/resnet50_iga.pt:0.627"
    )
    model_seed: int = 42
    #: Fallback only: a checkpoint that records its own input size wins over this value.
    model_input_size: int = 320

    enable_gradcam: bool = True

    #: Admission screening (POST /image/screen). `local` runs OpenCV YuNet + heuristics on this
    #: machine and rejects anything that is not a photograph of a human face; `gemini` delegates to
    #: the Gemini classifier when it is the active backend; `none` admits everything unchecked.
    screening_backend: Literal["local", "gemini", "none"] = "local"

    #: Acne gate in front of the IGA classifier (POST /infer/single): an image that does not show
    #: acne (freckles, vitiligo, other dermatoses, a healthy face) is answered `verdict: no_acne`
    #: and is NOT graded. Without the checkpoint the engine grades everything, as before, and
    #: reports `gate_checked: false`. The threshold defaults to the one stored in the checkpoint.
    acne_gate_enabled: bool = True
    acne_gate_checkpoint: str = "./checkpoints/acne_gate.pt"
    acne_gate_threshold: float | None = None

    #: Second opinion before grading (see `app/models/condition_check.py`): for every image the
    #: gate lets through as acne, an external model says whether it shows acne, healthy skin or
    #: another condition; «another condition» stops the grading (see below), anything else only
    #: warns. A failure never costs a grade. `none` turns it off; `gemini`
    #: needs `GEMINI_API_KEY` and sends a reduced copy of the photograph to Google's API.
    second_opinion_backend: Literal["none", "gemini"] = "none"
    #: When the second opinion says «another condition» (vitiligo, melasma…), the image is NOT
    #: graded — the same rule as the local gate's `no_acne`: an IGA grade for vitiligo is
    #: meaningless and would enter the record. `false` keeps it advisory (graded, with a warning).
    #: Healthy skin, acne next to something else, or «can't tell» are always graded.
    second_opinion_stops_grading: bool = True
    second_opinion_timeout_seconds: int = 20

    #: Gemini backend. The key lives in `.env`, which is gitignored; never in `.env.example`.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_seconds: int = 45

    tracking_driver: TrackingDriver = "file"
    tracking_dir: str = "./runs"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"

    data_dir: str = "./data"
    dataset_version: str = "v1.0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance, so the process reads the environment exactly once."""
    return Settings()
