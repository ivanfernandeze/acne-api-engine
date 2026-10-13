"""Trains the acne / healthy / other gate, picks its acne threshold and records everything.

Covers: Fase 4 de CRISP-DM para el filtro previo.

    uv run python -m training.gate.train_gate --manifest data/gate/manifest.csv

timm ``efficientnet_b0`` (ImageNet weights), three outputs in ``GATE_CLASSES`` order. The epoch
kept is the one with the best balanced accuracy on validation; the acne threshold is then the
HIGHEST probability cut that still keeps the sensitivity on acne at ``--target-sensitivity``
(0,97 by default). See ``app/models/acne_gate.py`` for how the two steps decide.

Writes ``checkpoints/acne_gate.candidate.pt`` — a CANDIDATE the engine does not load: it serves
only ``checkpoints/acne_gate.pt``, and the candidate is promoted (renamed) only after it has been
checked on real clinic photographs — and the full record of
the run — ``runs/gate/<run_id>/`` and ``acne-infra/docs/entrenamientos/filtro-acne-<run_id>.md``
(see ``training/gate/report.py``). Run ``training.gate.evaluate_gate`` afterwards to complete it
with the test results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.models.acne_gate import DEFAULT_INPUT_SIZE, GATE_CLASSES, decide, preprocess_image
from training.gate import report

ACNE = GATE_CLASSES.index("acne")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el filtro acné / piel sana / otra.")
    parser.add_argument("--manifest", default="data/gate/manifest.csv")
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--input-size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-sensitivity", type=float, default=0.97)
    parser.add_argument(
        "--output",
        default="checkpoints/acne_gate.candidate.pt",
        help=(
            "Candidato: el motor NO lo carga solo. Se activa renombrándolo a "
            "checkpoints/acne_gate.pt tras validarlo con fotos reales de la clínica."
        ),
    )
    parser.add_argument("--runs-dir", default="runs/gate")
    parser.add_argument("--docs-dir", default=str(report.DEFAULT_DOCS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4, help="Procesos de carga de imágenes.")
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Sin pesos ImageNet (solo pruebas rápidas)."
    )
    args = parser.parse_args()

    import timm
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    started_at = datetime.now(UTC)
    run_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = Path(args.manifest)
    rows = load_manifest(manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    for name, subset in (("train", train_rows), ("val", val_rows)):
        labels = {r["label"] for r in subset}
        if labels != set(GATE_CLASSES):
            raise SystemExit(f"El split {name} necesita las clases {GATE_CLASSES}; tiene {labels}.")

    # The dataset record travels with the run: the build report plus a copy of the manifest.
    dataset_record = report.read_json(manifest.with_name("dataset-report.json")) or {}
    dataset_record["manifest"] = manifest.as_posix()
    dataset_record["manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    report.write_json(run_dir / "dataset.json", dataset_record)
    shutil.copyfile(manifest, run_dir / "manifest.csv")

    train_loader = DataLoader(
        GateDataset(train_rows, args.input_size, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        GateDataset(val_rows, args.input_size, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    model = timm.create_model(
        args.model, pretrained=not args.no_pretrained, num_classes=len(GATE_CLASSES)
    )
    model.to(device)
    counts = np.bincount([GATE_CLASSES.index(r["label"]) for r in train_rows], minlength=3)
    class_weights = counts.sum() / (len(GATE_CLASSES) * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        probabilities, targets = predict(model, val_loader, device)
        metrics = gate_metrics(probabilities, targets, threshold=None)
        entry = {
            "epoch": epoch,
            "lr": round(lr, 8),
            "train_loss": round(total_loss / len(train_rows), 5),
            "val_balanced_accuracy": metrics["balanced_accuracy"],
            "val_acne_sensitivity": metrics["acne_sensitivity"],
            "val_acne_specificity": metrics["acne_specificity"],
            "val_recall_sana": metrics["recall"]["sana"],
            "val_recall_otra": metrics["recall"]["otra"],
            "val_acne_auc": metrics["acne_auc"],
        }
        history.append(entry)
        print(
            f"época {epoch:>2}  pérdida {entry['train_loss']:.4f}  "
            f"bal {metrics['balanced_accuracy']:.3f}"
            f"  sens.acné {metrics['acne_sensitivity']:.3f}  sana {metrics['recall']['sana']:.3f}"
            f"  otra {metrics['recall']['otra']:.3f}"
        )
        if best is None or metrics["balanced_accuracy"] > best["balanced_accuracy"]:
            best = {
                "epoch": epoch,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
    assert best is not None
    _write_csv(run_dir / "history.csv", history)

    model.load_state_dict(best["state"])
    probabilities, targets = predict(model, val_loader, device)
    threshold, curve = pick_threshold(probabilities, targets, args.target_sensitivity)
    _write_csv(run_dir / "threshold-curve.csv", curve)
    val_metrics = gate_metrics(probabilities, targets, threshold)
    report.write_json(run_dir / "metrics-val.json", val_metrics)
    print(
        f"\nÉpoca elegida {best['epoch']}; umbral de acné {threshold:.3f} -> sensibilidad "
        f"{val_metrics['acne_sensitivity']:.3f}, "
        f"especificidad {val_metrics['acne_specificity']:.3f}"
        f", exactitud balanceada {val_metrics['balanced_accuracy']:.3f} (val)"
    )

    version = f"gate3_{args.model}_{run_id}"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best["state"],
            "model_name": args.model,
            "classes": list(GATE_CLASSES),
            "input_size": args.input_size,
            "threshold": threshold,
            "version": version,
            "run_id": run_id,
            "seed": args.seed,
            "best_epoch": best["epoch"],
            "manifest_sha256": dataset_record["manifest_sha256"],
            # Plain JSON types only: torch.load(weights_only=True) refuses numpy scalars.
            "val_metrics": json.loads(json.dumps(val_metrics)),
        },
        output,
    )

    report.write_json(
        run_dir / "config.json",
        {
            "run_id": run_id,
            "version": version,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "arguments": vars(args),
            "classes": list(GATE_CLASSES),
            "class_counts_train": dict(zip(GATE_CLASSES, counts.tolist(), strict=True)),
            "class_weights": [round(float(w), 4) for w in class_weights],
            "best_epoch": best["epoch"],
            "threshold": threshold,
            "checkpoint": output.as_posix(),
            "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "environment": report.environment(),
        },
    )
    docs = report.write_report(run_dir, Path(args.docs_dir))
    print(f"Checkpoint: {output}\nRegistro: {run_dir}")
    if docs is not None:
        print(f"Reporte versionado: {docs}")
    print("Siguiente: python -m training.gate.evaluate_gate (completa el reporte con el test).")


# --- shared with evaluate_gate ----------------------------------------------------------------


def load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"No existe el manifiesto {path}. Corra training.gate.build_dataset.")
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_box(text: str) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    left, top, right, bottom = (int(v) for v in text.split(","))
    return left, top, right, bottom


class GateDataset:
    """Images from the manifest through ``preprocess_image`` — the serving preprocessing.

    The face box comes from the manifest (computed by the same ``face_box`` the service uses).
    Training augmentation never touches hue: horizontal flip; 30 % of the time HALF the face (one
    cheek), because CFD is frontal only and the protocol's profile shots must not read as acne just
    for being turned; a random square crop of 70-100 %; ±15 % brightness and contrast.
    """

    def __init__(self, rows: list[dict[str, str]], input_size: int, train: bool) -> None:
        self.rows = rows
        self.input_size = input_size
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        import torch
        from PIL import Image, ImageEnhance

        from app.models.acne_gate import square_around

        row = self.rows[index]
        with Image.open(row["path"]) as source:
            original_width = source.size[0]
            # JPEG DCT-domain downscaling: decoding a 2444x1718 CFD photograph in full, every
            # epoch, made training ten times slower for a picture that ends at 260 px anyway.
            source.draft("RGB", (DRAFT_SIDE, DRAFT_SIDE))
            image = source.convert("RGB")
        box = _scale_box(parse_box(row.get("face_box", "")), image.width / original_width)
        if self.train:
            if box is not None and random.random() < 0.3:
                left, top, right, bottom = box
                middle = (left + right) // 2
                overlap = (right - left) // 10
                box = (
                    (left, top, middle + overlap, bottom)
                    if random.random() < 0.5
                    else (middle - overlap, top, right, bottom)
                )
            image = square_around(image, box)
            box = None
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            side = int(min(image.size) * random.uniform(0.7, 1.0))
            left = random.randint(0, image.width - side)
            top = random.randint(0, image.height - side)
            image = image.crop((left, top, left + side, top + side))
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        array = preprocess_image(image, self.input_size, box)
        tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
        return tensor, GATE_CLASSES.index(row["label"])


DRAFT_SIDE = 1024


def _scale_box(
    box: tuple[int, int, int, int] | None, scale: float
) -> tuple[int, int, int, int] | None:
    if box is None or scale == 1:
        return box
    left, top, right, bottom = box
    return (int(left * scale), int(top * scale), int(right * scale), int(bottom * scale))


def predict(model: Any, loader: Any, device: Any) -> tuple[np.ndarray, np.ndarray]:
    """Softmax over the three classes per image, and the true class indices."""
    import torch

    model.eval()
    probabilities, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            probabilities.append(torch.softmax(model(images.to(device)), dim=1).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def decide_all(probabilities: np.ndarray, threshold: float | None) -> np.ndarray:
    """Class index per image; ``None`` = plain argmax (used while training)."""
    if threshold is None:
        return probabilities.argmax(axis=1)
    return np.array(
        [
            GATE_CLASSES.index(
                decide(dict(zip(GATE_CLASSES, map(float, row), strict=True)), threshold)
            )
            for row in probabilities
        ]
    )


def gate_metrics(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float | None
) -> dict[str, Any]:
    predicted = decide_all(probabilities, threshold)
    k = len(GATE_CLASSES)
    matrix = np.zeros((k, k), dtype=int)
    for true, pred in zip(targets, predicted, strict=True):
        matrix[true, pred] += 1
    recall = {c: _ratio(matrix[i, i], matrix[i].sum()) for i, c in enumerate(GATE_CLASSES)}
    precision = {c: _ratio(matrix[i, i], matrix[:, i].sum()) for i, c in enumerate(GATE_CLASSES)}
    is_acne = targets == ACNE
    predicted_acne = predicted == ACNE
    present = [v for c, v in recall.items() if matrix[GATE_CLASSES.index(c)].sum() > 0]
    return {
        "n": len(targets),
        "counts": {c: int((targets == i).sum()) for i, c in enumerate(GATE_CLASSES)},
        "threshold": None if threshold is None else round(float(threshold), 4),
        "acne_sensitivity": _ratio(int((predicted_acne & is_acne).sum()), int(is_acne.sum())),
        "acne_specificity": _ratio(int((~predicted_acne & ~is_acne).sum()), int((~is_acne).sum())),
        "acne_auc": round(auc(probabilities[:, ACNE], is_acne.astype(int)), 4),
        "balanced_accuracy": round(float(np.mean(present)), 4) if present else float("nan"),
        "recall": recall,
        "precision": precision,
        "classes": list(GATE_CLASSES),
        "confusion_matrix": matrix.tolist(),
    }


def auc(scores: np.ndarray, targets: np.ndarray) -> float:
    """ROC AUC by the rank (Mann-Whitney) formula, ties averaged; no scikit-learn needed."""
    positives, negatives = scores[targets == 1], scores[targets == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    values = np.concatenate([positives, negatives])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):
        tied = values == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    rank_sum = ranks[: len(positives)].sum()
    n_pos, n_neg = len(positives), len(negatives)
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


#: The acne threshold never exceeds this (see ``pick_threshold``).
MAX_THRESHOLD = 0.5


def pick_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_sensitivity: float,
    max_threshold: float = MAX_THRESHOLD,
) -> tuple[float, list[dict[str, float]]]:
    """Highest acne cut whose sensitivity is still ≥ the target, capped, plus the whole curve.

    The cap exists because a well-separated validation set saturates the acne scores near 1: the
    first run (20261010T205825Z) picked 0,998, and on test that kept 0,25 of the acne from another
    camera against 0,50 at 0,5, with the same specificity. A gate stricter than the argmax would be
    only buys fragility on photographs unlike the training ones.
    """
    acne_scores = probabilities[targets == ACNE, ACNE]
    others = probabilities[targets != ACNE, ACNE]
    curve = []
    for cut in np.round(np.linspace(0.01, 0.99, 99), 2):
        curve.append(
            {
                "threshold": float(cut),
                "acne_sensitivity": _ratio(int((acne_scores >= cut).sum()), len(acne_scores)),
                "acne_specificity": _ratio(int((others < cut).sum()), len(others)),
            }
        )
    if acne_scores.size == 0:
        return 0.5, curve
    keep = int(np.ceil(target_sensitivity * acne_scores.size))
    threshold = float(np.sort(acne_scores)[acne_scores.size - keep])
    # Truncated, never rounded: rounding up could leave the lowest kept acne score below the cut
    # and silently miss the sensitivity target. Probabilities are served with 4 decimals too.
    return min(math.floor(threshold * 1e4) / 1e4, max_threshold), curve


def _ratio(num: int, den: int) -> float:
    return round(float(num) / float(den), 4) if den else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
