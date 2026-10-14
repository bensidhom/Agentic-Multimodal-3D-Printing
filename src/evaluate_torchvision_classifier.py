"""Evaluate a saved torchvision classifier with group-bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from evaluate_yolo_classifier import group_intervals, load_source_groups, metric_vector, sha256
from train_torchvision_classifier import build_model, data_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--latency-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    data_root = args.data.resolve()
    manifest = args.dataset_manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_name = f"cuda:{args.device}" if str(args.device).isdigit() else str(args.device)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model, _ = build_model(checkpoint["architecture"], len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    _, evaluation_transform = data_transforms(int(checkpoint["image_size"]))
    dataset = datasets.ImageFolder(data_root / args.split, transform=evaluation_transform)
    if dataset.classes != classes:
        raise ValueError("Checkpoint and evaluation classes differ")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    probabilities: list[np.ndarray] = []
    y_true: list[int] = []
    inference_ms: list[float] = []
    started = perf_counter()
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_started = perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (perf_counter() - batch_started) * 1000.0 / targets.size(0)
            inference_ms.extend([elapsed] * targets.size(0))
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
            y_true.extend(targets.tolist())
    wall_seconds = perf_counter() - started

    # Match the YOLO evaluators' deployment-oriented latency endpoint: batch
    # one, including image loading, CPU transforms, host-to-device transfer,
    # model forward pass, and probability postprocessing.
    latency_rng = np.random.default_rng(args.seed)
    latency_count = min(args.latency_sample, len(dataset))
    latency_indices = latency_rng.choice(len(dataset), size=latency_count, replace=False)
    warm_image, _ = dataset[int(latency_indices[0])]
    warm_image = warm_image.unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(5):
            _ = torch.softmax(model(warm_image), dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
    end_to_end_ms: list[float] = []
    with torch.inference_mode():
        for index in latency_indices:
            if device.type == "cuda":
                torch.cuda.synchronize()
            sample_started = perf_counter()
            image, _ = dataset[int(index)]
            logits = model(image.unsqueeze(0).to(device, non_blocking=True))
            _ = torch.softmax(logits, dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_to_end_ms.append((perf_counter() - sample_started) * 1000.0)
    probability_array = np.vstack(probabilities)
    y_array = np.asarray(y_true, dtype=int)
    group_lookup = load_source_groups(manifest, args.split)
    rows: list[dict[str, object]] = []
    groups: list[str] = []
    for (path_string, target), probs in zip(dataset.samples, probability_array, strict=True):
        path = Path(path_string).resolve()
        digest_key = "sha256:" + sha256(path).lower()
        path_key = "path:" + "/".join(path.parts[-3:]).lower()
        group = group_lookup.get(digest_key, group_lookup.get(path_key, path_key))
        groups.append(group)
        predicted = int(probs.argmax())
        row: dict[str, object] = {
            "path": str(path),
            "source_group_id": group,
            "true_class": classes[target],
            "predicted_class": classes[predicted],
            "confidence": float(probs[predicted]),
        }
        row.update({f"prob_{name}": float(probs[index]) for index, name in enumerate(classes)})
        rows.append(row)
    group_array = np.asarray(groups)
    metrics = {
        "schema_version": 1,
        "purpose": "torchvision_classification_benchmark",
        "model": {
            "architecture": checkpoint["architecture"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_size_mb": checkpoint_path.stat().st_size / 1_000_000,
            "selected_epoch": checkpoint["epoch"],
        },
        "dataset": {
            "root": str(data_root),
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "split": args.split,
            "images": len(dataset),
            "source_groups": len(set(groups)),
            "classes": classes,
        },
        "metrics": metric_vector(y_array, probability_array),
        "confidence_intervals": group_intervals(y_array, probability_array, group_array, args.bootstrap, args.seed),
        "latency": {
            "measurement": "Batch-one end-to-end wall time after warm-up, CUDA-synchronized; includes file input, CPU transforms, host-to-device transfer, inference, and probability postprocessing.",
            "sample_images": latency_count,
            "median_ms": float(median(end_to_end_ms)),
            "p95_ms": float(np.quantile(end_to_end_ms, 0.95)),
            "mean_batched_forward_ms_per_image": float(np.mean(inference_ms)),
            "prediction_pass_wall_seconds": wall_seconds,
        },
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": metrics["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
