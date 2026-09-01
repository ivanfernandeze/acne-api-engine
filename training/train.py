"""Fine-tune EfficientNet-B2 on the IGA scale.

Covers: Fase 4 de CRISP-DM (seccion 8.1). Run with:

    uv run python -m training.train --config training/config.yaml

Requires the ``[torch]`` extra and a prepared dataset. Every run writes its hyperparameters,
per-epoch metrics and the winning checkpoint under ``tracking_dir``, so the result is auditable
without MLflow; set ``TRACKING_DRIVER=mlflow`` to mirror the same records into a server.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.domain.iga import IGA_CLASSES
from training.dataset import build_torch_dataset, class_weights, load_manifest, stratified_split


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena EfficientNet-B2 sobre la escala IGA.")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--manifest", default=None, help="CSV con path,iga_class[,age,fitzpatrick]")
    args = parser.parse_args()

    config: dict[str, Any] = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    try:
        import timm
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise SystemExit(
            "Falta el extra [torch]. Instale con:\n"
            "  uv pip install -e '.[torch]'\n"
            "En la RTX 5060 (Blackwell, sm_120) use CUDA 12.8 o superior:\n"
            "  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
        ) from error

    seed = int(config["seed"])
    set_global_seed(seed)

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

    input_size = int(config["model"]["input_size"])
    batch_size = int(config["training"]["batch_size"])

    train_loader = DataLoader(
        build_torch_dataset(splits["train"], input_size, train=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        build_torch_dataset(splits["val"], input_size, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    model = timm.create_model(
        config["model"]["name"],
        pretrained=bool(config["model"]["pretrained"]),
        num_classes=len(IGA_CLASSES),
    ).to(device)

    _freeze_backbone(model, int(config["model"]["unfreeze_blocks"]))

    weights = torch.tensor(class_weights(splits["train"]), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=float(config["training"]["label_smoothing"])
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    patience = int(config["training"]["early_stopping"]["patience"])
    best_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(config["tracking_dir"]) / config["dataset_version"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(config["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_f1 = _validate(model, val_loader, criterion, device)
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
        history.append(record)
        print(
            f"epoch {epoch:03d}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}  val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "version": f"{config['model']['name']}_{run_id}",
                    "seed": seed,
                    "epoch": epoch,
                    "val_f1_macro": val_f1,
                    "classes": list(IGA_CLASSES),
                    "dataset_version": config["dataset_version"],
                },
                checkpoint_dir / "efficientnet_b2_iga.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping en la época {epoch}: {patience} épocas sin mejora.")
                break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\nMejor F1 macro de validación: {best_f1:.4f}")
    print(f"Checkpoint: {checkpoint_dir / 'efficientnet_b2_iga.pt'}")
    print(f"Registro de la corrida: {run_dir}")
    print(
        "\nCriterio de éxito de la Fase 1: F1 macro >= 0,752 sobre el test congelado "
        "(85 % del 0,885 reportado por Watanabe, 2025)."
    )
    print("Evalúe ahora con:  uv run python -m training.evaluate")


def _freeze_backbone(model: Any, unfreeze_blocks: int) -> None:
    """Transfer learning: train only the head plus the last convolutional blocks (seccion 8.1)."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable: list[Any] = [model.get_classifier()]
    blocks = getattr(model, "blocks", None)
    if blocks is not None and unfreeze_blocks > 0:
        trainable.extend(blocks[-unfreeze_blocks:])
    conv_head = getattr(model, "conv_head", None)
    if conv_head is not None:
        trainable.append(conv_head)

    for module in trainable:
        for parameter in module.parameters():
            parameter.requires_grad = True


def _run_epoch(model: Any, loader: Any, criterion: Any, optimizer: Any, device: Any) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


def _validate(model: Any, loader: Any, criterion: Any, device: Any) -> tuple[float, float, float]:
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
            logits = model(images)
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
