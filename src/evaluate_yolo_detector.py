"""Evaluate a YOLO detector on localization and shared image-level endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PLATFORM_ROOT / ".ultralytics"))

from evaluate_yolo_classifier import group_intervals, metric_vector, sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--latency-sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_truth(manifest: Path, split: str) -> dict[str, dict[str, str]]:
    truth: dict[str, dict[str, str]] = {}
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("split") != split:
                continue
            truth[row["image_sha256"].lower()] = {
                "target": row["classification_target"],
                "group": row["source_group_id"],
            }
    return truth


def image_paths(data_yaml: Path, split: str) -> list[Path]:
    import yaml

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    entry = Path(config[split])
    if not entry.is_absolute():
        entry = (root / entry).resolve()
    return sorted(
        path
        for path in entry.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def end_to_end_latency(
    model: object, images: list[Path], sample_size: int, imgsz: int, device: str
) -> list[float]:
    import torch

    if sample_size <= 0:
        return []
    indices = np.linspace(0, len(images) - 1, min(sample_size, len(images)), dtype=int)
    values: list[float] = []
    for index in indices:
        if torch.cuda.is_available() and str(device) != "cpu":
            torch.cuda.synchronize()
        started = perf_counter()
        model.predict(
            source=str(images[index]),
            imgsz=imgsz,
            conf=0.001,
            device=device,
            verbose=False,
        )
        if torch.cuda.is_available() and str(device) != "cpu":
            torch.cuda.synchronize()
        values.append((perf_counter() - started) * 1000.0)
    return values


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_yaml = args.data_yaml.resolve()
    manifest = args.dataset_manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file() or not data_yaml.is_file() or not manifest.is_file():
        raise FileNotFoundError("Model, data YAML, or dataset manifest is missing")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    detection_names = [model.names[index] for index in sorted(model.names)]
    shared_names = ["background", *detection_names]
    name_to_index = {name: index for index, name in enumerate(shared_names)}
    images = image_paths(data_yaml, args.split)
    truth = load_truth(manifest, args.split)
    model.predict(source=str(images[0]), imgsz=args.imgsz, device=args.device, verbose=False)

    prediction_started = perf_counter()
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=0.001,
        iou=0.7,
        batch=args.batch,
        device=args.device,
        stream=True,
        verbose=False,
    )
    rows: list[dict[str, object]] = []
    y_true: list[int] = []
    probabilities: list[np.ndarray] = []
    groups: list[str] = []
    model_inference: list[float] = []
    for path, result in zip(images, results, strict=True):
        digest = sha256(path).lower()
        if digest not in truth:
            raise KeyError(f"Test image absent from frozen manifest: {path.name}")
        target = truth[digest]["target"]
        class_scores = np.zeros(len(detection_names), dtype=float)
        if result.boxes is not None and len(result.boxes):
            confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            for class_id, confidence in zip(class_ids, confidences):
                class_scores[class_id] = max(class_scores[class_id], confidence)
        max_score = float(class_scores.max(initial=0.0))
        # The decision threshold occupies the background score slot. This makes
        # argmax exactly reproduce: no class above threshold => background.
        scores = np.concatenate(([args.confidence], class_scores))
        scores /= scores.sum() if scores.sum() else 1.0
        predicted_index = 0 if max_score < args.confidence else int(class_scores.argmax()) + 1
        predicted_name = shared_names[predicted_index]
        row: dict[str, object] = {
            "path": str(path),
            "image_sha256": digest,
            "source_group_id": truth[digest]["group"],
            "true_class": target,
            "predicted_class": predicted_name,
            "max_detection_confidence": max_score,
            "detections_at_0_001": int(len(result.boxes)) if result.boxes is not None else 0,
            "model_reported_inference_ms": float(result.speed.get("inference", np.nan)),
        }
        row.update({f"score_{name}": float(scores[index]) for index, name in enumerate(shared_names)})
        rows.append(row)
        y_true.append(name_to_index[target])
        probabilities.append(scores)
        groups.append(truth[digest]["group"])
        model_inference.append(float(result.speed.get("inference", np.nan)))
    prediction_seconds = perf_counter() - prediction_started

    y_array = np.asarray(y_true, dtype=int)
    probability_array = np.vstack(probabilities)
    group_array = np.asarray(groups)
    shared_metrics = metric_vector(y_array, probability_array)
    latency = np.asarray(
        end_to_end_latency(model, images, args.latency_sample, args.imgsz, args.device)
    )

    validation = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        save_json=False,
        project=str(output),
        name="localization_evaluation",
        exist_ok=True,
        verbose=False,
    )
    box = validation.box
    per_class = {}
    for index, name in enumerate(detection_names):
        per_class[name] = {
            "precision": float(box.p[index]),
            "recall": float(box.r[index]),
            "ap50": float(box.ap50[index]),
            "ap50_95": float(box.maps[index]),
        }

    metrics = {
        "schema_version": 1,
        "purpose": "pipeline_smoke_test" if args.smoke_test else "detection_benchmark",
        "scientific_use": (
            "Pipeline verification only; do not report as a paper result."
            if args.smoke_test
            else "Frozen-split evaluation; threshold and calibration must have been fixed on validation data."
        ),
        "model": {
            "path": str(model_path),
            "sha256": sha256(model_path),
            "size_mb": model_path.stat().st_size / 1_000_000,
        },
        "dataset": {
            "data_yaml": str(data_yaml),
            "data_yaml_sha256": sha256(data_yaml),
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "split": args.split,
            "images": len(images),
            "source_groups": len(set(groups)),
            "shared_classes": shared_names,
        },
        "shared_image_level_metrics": shared_metrics,
        "shared_metric_confidence_intervals": group_intervals(
            y_array, probability_array, group_array, args.bootstrap, args.seed
        ),
        "localization_metrics": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "ap50": float(box.map50),
            "ap50_95": float(box.map),
            "per_class": per_class,
        },
        "score_note": "Shared image-level scores are an uncalibrated decision-score normalization: the validation-selected confidence threshold is the background score and each defect score is the maximum box confidence for that class. Treat calibration metrics as diagnostic unless a calibration model is fit on validation data and frozen before test evaluation.",
        "latency": {
            "measurement": "Batch-one end-to-end wall time after warm-up, CUDA-synchronized; includes file input, preprocessing, inference, and postprocessing.",
            "sample_images": int(latency.size),
            "median_ms": float(median(latency)) if latency.size else None,
            "p95_ms": float(np.quantile(latency, 0.95)) if latency.size else None,
            "mean_model_reported_inference_ms": float(np.nanmean(model_inference)),
            "prediction_pass_wall_seconds": prediction_seconds,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "shared_image_level_metrics": shared_metrics,
                "localization_metrics": metrics["localization_metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
