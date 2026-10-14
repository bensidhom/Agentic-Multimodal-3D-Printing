"""Evaluate an Ultralytics classifier on a frozen split.

Outputs row-level probabilities plus paper-ready aggregate metrics. Confidence
intervals resample source groups rather than individual frames when a compatible
dataset manifest is supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PLATFORM_ROOT / ".ultralytics"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--latency-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, bins: int = 15
) -> float:
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(value)


def metric_vector(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    classes = np.arange(probabilities.shape[1])
    matrix = confusion_matrix(y_true, predicted, labels=classes)
    specificity = []
    for index in classes:
        true_negative = matrix.sum() - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
        false_positive = matrix[:, index].sum() - matrix[index, index]
        denominator = true_negative + false_positive
        specificity.append(true_negative / denominator if denominator else np.nan)
    one_hot = np.eye(probabilities.shape[1], dtype=float)[y_true]
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_precision": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_recall_sensitivity": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
        "macro_specificity_ovr": float(np.nanmean(specificity)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "expected_calibration_error_15bin": expected_calibration_error(y_true, probabilities),
    }


def group_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    unique = np.unique(groups)
    if iterations <= 0 or len(unique) < 2:
        return {}
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([rows_by_group[group] for group in chosen])
        # Skip replicates missing a class: they do not estimate the frozen
        # multiclass endpoint and can make balanced metrics misleading.
        if len(np.unique(y_true[indices])) != probabilities.shape[1]:
            continue
        for name, value in metric_vector(y_true[indices], probabilities[indices]).items():
            samples[name].append(value)
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "valid_replicates": len(values),
        }
        for name, values in samples.items()
        if values
    }


def load_source_groups(manifest_path: Path, split: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            row_split = row.get("partition") or row.get("split")
            if row_split != split:
                continue
            path_value = (
                row.get("classification_path")
                or row.get("processed_path")
                or row.get("image_path")
            )
            if not path_value:
                continue
            path = Path(path_value)
            group = (
                row.get("source_group_id")
                or row.get("augmentation_group_id")
                or row.get("record_id")
                or path.name
            )
            mapping["path:" + "/".join(path.parts[-3:]).lower()] = group
            digest = row.get("processed_sha256") or row.get("image_sha256")
            if digest:
                mapping["sha256:" + digest.lower()] = group
    return mapping


def measure_end_to_end_latency(
    model: object,
    images: list[Path],
    sample_size: int,
    imgsz: int,
    device: str,
) -> list[float]:
    """Measure batch-one wall latency after warm-up, including I/O and NMS/head work."""
    if sample_size <= 0:
        return []
    import torch

    count = min(sample_size, len(images))
    indices = np.linspace(0, len(images) - 1, count, dtype=int)
    samples: list[float] = []
    for index in indices:
        if torch.cuda.is_available() and str(device) != "cpu":
            torch.cuda.synchronize()
        started = perf_counter()
        model.predict(source=str(images[index]), imgsz=imgsz, device=device, verbose=False)
        if torch.cuda.is_available() and str(device) != "cpu":
            torch.cuda.synchronize()
        samples.append((perf_counter() - started) * 1000.0)
    return samples


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_root = args.data.resolve()
    manifest_path = args.dataset_manifest.resolve()
    split_root = data_root / args.split
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file() or not split_root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("Model, split root, or dataset manifest is missing")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = [model.names[index] for index in sorted(model.names)]
    class_to_index = {name: index for index, name in enumerate(names)}
    images = sorted(
        path
        for path in split_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    unknown = sorted({path.parent.name for path in images} - set(class_to_index))
    if unknown:
        raise ValueError(f"Dataset classes absent from checkpoint: {unknown}")

    # One warm-up call prevents model initialization from contaminating latency.
    model.predict(source=str(images[0]), imgsz=args.imgsz, device=args.device, verbose=False)
    wall_start = perf_counter()
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        stream=True,
        verbose=False,
    )

    group_lookup = load_source_groups(manifest_path, args.split)
    rows: list[dict[str, object]] = []
    probabilities: list[np.ndarray] = []
    y_true: list[int] = []
    latencies: list[float] = []
    # Ultralytics names in-memory/list inputs image0.jpg, image1.jpg, ...;
    # retain the ordered source list as the authoritative path mapping.
    for path, result in zip(images, results, strict=True):
        path = path.resolve()
        true_name = path.parent.name
        true_index = class_to_index[true_name]
        probs = result.probs.data.detach().cpu().numpy().astype(float)
        probs /= probs.sum()
        predicted_index = int(probs.argmax())
        path_key = "path:" + "/".join(path.parts[-3:]).lower()
        digest_key = "sha256:" + sha256(path).lower()
        group = group_lookup.get(digest_key, group_lookup.get(path_key, path_key))
        latency = float(result.speed.get("inference", np.nan))
        row: dict[str, object] = {
            "path": str(path),
            "source_group_id": group,
            "true_class": true_name,
            "true_index": true_index,
            "predicted_class": names[predicted_index],
            "predicted_index": predicted_index,
            "confidence": float(probs[predicted_index]),
            "inference_ms": latency,
        }
        row.update({f"prob_{name}": float(probs[index]) for index, name in enumerate(names)})
        rows.append(row)
        probabilities.append(probs)
        y_true.append(true_index)
        latencies.append(latency)
    wall_seconds = perf_counter() - wall_start

    y_array = np.asarray(y_true, dtype=int)
    probability_array = np.vstack(probabilities)
    predicted = probability_array.argmax(axis=1)
    groups = np.asarray([row["source_group_id"] for row in rows])
    base_metrics = metric_vector(y_array, probability_array)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_array,
        predicted,
        labels=np.arange(len(names)),
        zero_division=0,
    )
    matrix = confusion_matrix(y_array, predicted, labels=np.arange(len(names)))
    finite_latency = np.asarray([value for value in latencies if np.isfinite(value)])
    end_to_end_latency = np.asarray(
        measure_end_to_end_latency(
            model, images, args.latency_sample, args.imgsz, args.device
        )
    )

    metrics = {
        "schema_version": 1,
        "purpose": "pipeline_smoke_test" if args.smoke_test else "classification_benchmark",
        "scientific_use": (
            "Pipeline verification only; do not report as a paper result."
            if args.smoke_test
            else "Frozen-split evaluation; interpret confidence intervals with source-group limitations."
        ),
        "model": {"path": str(model_path), "sha256": sha256(model_path), "size_mb": model_path.stat().st_size / 1_000_000},
        "dataset": {
            "root": str(data_root),
            "split": args.split,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "images": len(images),
            "source_groups": int(len(np.unique(groups))),
            "classes": names,
        },
        "metrics": base_metrics,
        "confidence_intervals": group_intervals(
            y_array, probability_array, groups, args.bootstrap, args.seed
        ),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(names)
        },
        "confusion_matrix": matrix.tolist(),
        "latency": {
            "measurement": "Batch-one end-to-end wall time after warm-up, CUDA-synchronized; includes file input, preprocessing, inference, and postprocessing.",
            "sample_images": int(end_to_end_latency.size),
            "median_ms": float(median(end_to_end_latency)) if end_to_end_latency.size else None,
            "p95_ms": float(np.quantile(end_to_end_latency, 0.95)) if end_to_end_latency.size else None,
            "mean_model_reported_inference_ms": float(finite_latency.mean()) if finite_latency.size else None,
            "prediction_pass_wall_seconds": wall_seconds,
        },
        "false_alarm_note": (
            "The background class permits image-level false-positive and one-versus-rest "
            "specificity estimates. False alarms per print hour still require time-indexed, "
            "independent print runs."
            if "background" in names
            else "A normal/no-defect class is absent, so operational false alarms and "
            "specificity versus normal prints cannot be estimated from this dataset."
        ),
        "arguments": vars(args) | {"model": str(args.model), "data": str(args.data), "dataset_manifest": str(args.dataset_manifest), "output": str(args.output)},
    }

    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": base_metrics}, indent=2))


if __name__ == "__main__":
    main()
