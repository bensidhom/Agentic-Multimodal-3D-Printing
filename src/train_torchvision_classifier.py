"""Train provenance-recorded torchvision classifiers on frozen folder splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("resnet18", "mobilenet_v3_small", "efficientnet_b0"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_model(name: str, classes: int, pretrained: bool) -> tuple[nn.Module, str]:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, classes)
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, classes)
    else:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, classes)
    return model, str(weights)


def data_transforms(size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(int(round(size * 256 / 224))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train, evaluation


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def validation_pass(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float, float, float]:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            losses.append(float(criterion(logits, targets).item()) * targets.size(0))
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    count = len(y_true)
    return (
        sum(losses) / count,
        float(accuracy_score(y_true, y_pred)),
        float(balanced_accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch_cache = ROOT / ".torch_cache"
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(torch_cache))
    data_root = args.data.resolve()
    manifest = args.dataset_manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not manifest.is_file() or not (data_root / "train").is_dir() or not (data_root / "val").is_dir():
        raise FileNotFoundError("Frozen data root or manifest is missing")
    device_name = f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    train_transform, evaluation_transform = data_transforms(args.imgsz)
    train_data = datasets.ImageFolder(data_root / "train", transform=train_transform)
    val_data = datasets.ImageFolder(data_root / "val", transform=evaluation_transform)
    if train_data.class_to_idx != val_data.class_to_idx:
        raise ValueError("Train and validation class mappings differ")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    started = datetime.now(timezone.utc).isoformat()
    manifest_record = {
        "schema_version": 1,
        "started_utc": started,
        "purpose": "torchvision_classification_benchmark",
        "scientific_use": "Candidate benchmark; dataset limitations govern interpretation.",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset": {"root": str(data_root), "manifest": str(manifest), "manifest_sha256": sha256(manifest)},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "hardware": {
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "git_revision": git_revision(),
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest_record, indent=2), encoding="utf-8")
    try:
        model, weights_name = build_model(args.model, len(train_data.classes), not args.no_pretrained)
        model.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best_f1 = -1.0
        best_epoch = 0
        stale = 0
        rows: list[dict[str, float | int]] = []
        run_start = perf_counter()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            sample_count = 0
            for images, targets in train_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(images)
                    loss = criterion(logits, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.item()) * targets.size(0)
                sample_count += targets.size(0)
            val_loss, accuracy, balanced, macro_f1 = validation_pass(model, val_loader, criterion, device)
            row = {
                "epoch": epoch,
                "elapsed_seconds": perf_counter() - run_start,
                "train_loss": loss_sum / sample_count,
                "val_loss": val_loss,
                "val_accuracy": accuracy,
                "val_balanced_accuracy": balanced,
                "val_macro_f1": macro_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            rows.append(row)
            with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            checkpoint = {
                "schema_version": 1,
                "architecture": args.model,
                "weights": weights_name,
                "classes": train_data.classes,
                "class_to_idx": train_data.class_to_idx,
                "image_size": args.imgsz,
                "epoch": epoch,
                "model_state": model.state_dict(),
            }
            torch.save(checkpoint, output / "last.pt")
            if macro_f1 > best_f1 + 1e-12:
                best_f1 = macro_f1
                best_epoch = epoch
                stale = 0
                torch.save(checkpoint, output / "best.pt")
            else:
                stale += 1
            print(
                f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
                f"val_accuracy={accuracy:.4f} val_macro_f1={macro_f1:.4f} stale={stale}/{args.patience}",
                flush=True,
            )
            scheduler.step()
            if stale >= args.patience:
                break
        manifest_record.update(
            {
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "pretrained_weights": weights_name,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
                "best_checkpoint": str((output / "best.pt").resolve()),
            }
        )
    except Exception as error:
        manifest_record.update(
            {
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        raise
    finally:
        (output / "run_manifest.json").write_text(json.dumps(manifest_record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
