"""Fine-tune EfficientNet-B2 on the IGA scale.

Covers: Fase 4 de CRISP-DM (seccion 8.1). Run with:

    uv run python -m training.train --config training/config.yaml

Requires the ``[torch]`` extra and a prepared dataset. Every run writes its hyperparameters,
per-epoch metrics and the winning checkpoint under ``tracking_dir``, so the result is auditable
without MLflow; set ``TRACKING_DRIVER=mlflow`` to mirror the same records into a server.

``fit`` holds the whole recipe so ``training.cross_validate`` trains every fold with exactly the
same code path as the final model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.domain.iga import IGA_CLASSES
from training.dataset import (
    Sample,
    balanced_sampler,
    build_torch_dataset,
    class_weights,
    load_manifest,
    stratified_split,
)


def set_global_seed(seed: int) -> None:
    """Pin every source of randomness (RNF-03).

    ``cudnn.deterministic`` costs some throughput and is enabled anyway: a thesis result that
    cannot be reproduced is not a result.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class FitResult:
    """What one training run leaves behind."""

    best_f1: float
    best_epoch: int
    best_state: dict[str, Any]
    history: list[dict[str, float]] = field(default_factory=list)


def fit(
    config: dict[str, Any],
    train_samples: list[Sample],
    val_samples: list[Sample],
    device: Any,
    seed: int,
    log_prefix: str = "",
) -> FitResult:
    """Train on ``train_samples`` selecting the epoch with the best macro F1 on ``val_samples``."""
    import timm
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    set_global_seed(seed)

    model_cfg = config["model"]
    train_cfg = config["training"]
    aug_cfg = config.get("augmentation", {})

    input_size = int(model_cfg["input_size"])
    batch_size = int(train_cfg["batch_size"])
    crop_scale = aug_cfg.get("random_resized_crop_scale")

    use_sampler = bool(train_cfg.get("balanced_sampler", False))
    train_loader = DataLoader(
        build_torch_dataset(
            train_samples,
            input_size,
            train=True,
            crop_scale=tuple(crop_scale) if crop_scale else None,
        ),
        batch_size=batch_size,
        shuffle=not use_sampler,
        sampler=balanced_sampler(train_samples, seed) if use_sampler else None,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        build_torch_dataset(val_samples, input_size, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = timm.create_model(
        model_cfg["name"],
        pretrained=bool(model_cfg["pretrained"]),
        num_classes=len(IGA_CLASSES),
        drop_rate=float(model_cfg.get("dropout", 0.3)),
    ).to(device)

    unfreeze_blocks = model_cfg["unfreeze_blocks"]
    _freeze_backbone(model, unfreeze_blocks)

    # Resampling already balances what the network sees; weighting the loss on top of it would
    # count the rare classes twice and over-predict `severo`.
    weights = (
        None
        if use_sampler or train_cfg.get("class_weight", "balanced") != "balanced"
        else torch.tensor(class_weights(train_samples), dtype=torch.float32, device=device)
    )
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=float(train_cfg["label_smoothing"])
    )

    optimizer = torch.optim.AdamW(
        _param_groups(model, train_cfg),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    epochs = int(train_cfg["epochs"])
    scheduler = _build_scheduler(optimizer, epochs, int(train_cfg.get("warmup_epochs", 0)))

    use_amp = bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda"
    cutmix_probability = float(aug_cfg.get("cutmix_probability", 0.0))

    patience = int(train_cfg["early_stopping"]["patience"])
    best = FitResult(best_f1=-1.0, best_epoch=0, best_state={})
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = _run_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, cutmix_probability
        )
        val_loss, val_accuracy, val_f1 = _validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_accuracy, 6),
            "val_f1_macro": round(val_f1, 6),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.perf_counter() - started, 2),
        }
        best.history.append(record)
        print(
            f"{log_prefix}epoch {epoch:03d}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}  val_f1={val_f1:.4f}"
        )

        if val_f1 > best.best_f1:
            best.best_f1 = val_f1
            best.best_epoch = epoch
            best.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"{log_prefix}Early stopping en la época {epoch}: {patience} épocas sin mejora."
                )
                break

    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena un clasificador timm (escala IGA).")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--manifest", default=None, help="CSV con path,iga_class[,age,fitzpatrick]")
    parser.add_argument(
        "--model",
        default=None,
        help="Nombre timm; sobrescribe model.name del config (p. ej. resnet50)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Nombre del checkpoint (checkpoints/<tag>_iga.pt). Por defecto, el nombre del modelo.",
    )
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model is not None:
        config["model"]["name"] = args.model
    tag = args.tag or str(config["model"]["name"])

    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "Falta el extra [torch]. Instale con:\n"
            "  uv pip install -e '.[torch]'\n"
            "En la RTX 5060 (Blackwell, sm_120) use CUDA 12.8 o superior:\n"
            "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        ) from error

    seed = int(config["seed"])

    manifest = Path(args.manifest or Path(config["data_dir"]) / "manifest.csv")
    if not manifest.is_file():
        raise SystemExit(
            f"No se encontró el manifiesto {manifest}. Ejecute antes la Fase 3:\n"
            "  uv run python -m training.preprocess"
        )

    samples = load_manifest(manifest)
    splits = stratified_split(
        samples,
        (config["splits"]["train"], config["splits"]["val"], config["splits"]["test"]),
        seed,
    )
    print(
        f"Muestras: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["tracking_dir"]) / config["dataset_version"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{tag}_iga.pt"

    result = fit(config, splits["train"], splits["val"], device, seed)

    torch.save(
        {
            "state_dict": result.best_state,
            # `model_name` is what the inference backend feeds to `timm.create_model`; without
            # it a checkpoint is assumed to be the EfficientNet-B2 of section 8.1.
            "model_name": config["model"]["name"],
            "version": f"{config['model']['name']}_{run_id}",
            "seed": seed,
            "epoch": result.best_epoch,
            "val_f1_macro": result.best_f1,
            "classes": list(IGA_CLASSES),
            "dataset_version": config["dataset_version"],
            "input_size": int(config["model"]["input_size"]),
        },
        checkpoint_path,
    )

    (run_dir / "history.json").write_text(json.dumps(result.history, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\nMejor F1 macro de validación: {result.best_f1:.4f} (época {result.best_epoch})")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Registro de la corrida: {run_dir}")
    print(
        "\nCriterio de éxito de la Fase 1: F1 macro >= 0,752 sobre el test congelado "
        "(85 % del 0,885 reportado por Watanabe, 2025)."
    )
    print("Evalúe ahora con:  uv run python -m training.evaluate")


def _freeze_backbone(model: Any, unfreeze_blocks: int | str) -> None:
    """Transfer learning (seccion 8.1).

    ``unfreeze_blocks: all`` trains the whole network (the backbone at its own, lower learning
    rate, see ``_param_groups``); an integer trains only the head plus that many trailing
    convolutional blocks.
    """
    if unfreeze_blocks == "all":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable: list[Any] = [model.get_classifier()]
    blocks = getattr(model, "blocks", None)
    if blocks is not None and int(unfreeze_blocks) > 0:
        trainable.extend(blocks[-int(unfreeze_blocks) :])
    conv_head = getattr(model, "conv_head", None)
    if conv_head is not None:
        trainable.append(conv_head)

    for module in trainable:
        for parameter in module.parameters():
            parameter.requires_grad = True


def _param_groups(model: Any, train_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Head at ``learning_rate``, pretrained backbone at ``backbone_learning_rate`` (if set).

    ImageNet features are a good prior for skin texture; moving them with the same step as a
    freshly initialised head destroys them in the first epochs.
    """
    head_lr = float(train_cfg["learning_rate"])
    backbone_lr = float(train_cfg.get("backbone_learning_rate", head_lr))
    head_params = {id(p) for p in model.get_classifier().parameters()}

    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_params]
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_params]
    groups: list[dict[str, Any]] = [{"params": head, "lr": head_lr}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr})
    return groups


def _build_scheduler(optimizer: Any, epochs: int, warmup_epochs: int) -> Any:
    """Linear warmup followed by cosine annealing to zero."""
    import torch

    def factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _cutmix(images: Any, labels: Any) -> tuple[Any, Any, Any, float]:
    """CutMix (Yun et al., 2019): paste a random box from a shuffled batch, mix the labels.

    Boxes are drawn with ``torch`` so the augmentation follows the global seed.
    """
    import torch

    batch = images.size(0)
    lam = float(torch.distributions.Beta(1.0, 1.0).sample())
    permutation = torch.randperm(batch, device=images.device)

    height, width = images.shape[-2:]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    cy = int(torch.randint(height, (1,)))
    cx = int(torch.randint(width, (1,)))
    y1, y2 = max(0, cy - cut_h // 2), min(height, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(width, cx + cut_w // 2)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / float(height * width)
    return mixed, labels, labels[permutation], lam


def _run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    use_amp: bool,
    cutmix_probability: float,
) -> float:
    import torch

    model.train()
    total_loss = 0.0
    batches = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        apply_cutmix = cutmix_probability > 0 and float(torch.rand(1)) < cutmix_probability
        if apply_cutmix:
            images, labels_a, labels_b, lam = _cutmix(images, labels)

        # bfloat16 keeps the fp32 exponent range, so no GradScaler is needed (unlike float16).
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(images)
            if apply_cutmix:
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


def _validate(
    model: Any, loader: Any, criterion: Any, device: Any, use_amp: bool = False
) -> tuple[float, float, float]:
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    total_loss = 0.0
    batches = 0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images)
            logits = logits.float()
            total_loss += float(criterion(logits, labels).item())
            batches += 1
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    accuracy = (
        sum(int(p == t) for p, t in zip(predictions, targets, strict=True)) / len(targets)
        if targets
        else 0.0
    )
    f1 = float(f1_score(targets, predictions, average="macro", zero_division=0)) if targets else 0.0
    return total_loss / max(batches, 1), accuracy, f1


if __name__ == "__main__":
    main()
