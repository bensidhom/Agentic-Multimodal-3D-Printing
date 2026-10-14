"""Select a detector-to-image-classification threshold on a non-test split.

The detector emits zero or more localized defect boxes. This adapter converts
the maximum confidence observed for each defect class into the shared
three-class endpoint used by the image classifier: background,
spaghettification, and underextrusion. The threshold is selected only from a
training or validation split and must then be passed unchanged to test
evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PLATFORM_ROOT / ".ultralytics"))

from evaluate_yolo_classifier import group_intervals, metric_vector, sha256  # noqa: E402
from evaluate_yolo_detector import image_paths, load_truth  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-steps", type=int, default=99)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def shared_probabilities(class_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Create normalized comparison scores with threshold as background score."""
    if class_scores.ndim != 2:
        raise ValueError("class_scores must be a two-dimensional array")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    scores = np.column_stack(
        (np.full(class_scores.shape[0], threshold, dtype=float), class_scores)
    )
    return scores / scores.sum(axis=1, keepdims=True)


def select_threshold(
    y_true: np.ndarray, class_scores: np.ndarray, thresholds: np.ndarray
) -> tuple[float, list[dict[str, float]]]:
    """Maximize macro F1, then balanced accuracy, on an ordered threshold grid."""
    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = metric_vector(y_true, shared_probabilities(class_scores, float(threshold)))
        curve.append({"threshold": float(threshold), **metrics})
    selected = max(
        curve,
        key=lambda row: (
            row["macro_f1"],
            row["balanced_accuracy"],
            -row["threshold"],
        ),
    )
    return float(selected["threshold"]), curve


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    data_yaml = args.data_yaml.resolve()
    manifest = args.dataset_manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file() or not data_yaml.is_file() or not manifest.is_file():
        raise FileNotFoundError("Model, data YAML, or dataset manifest is missing")
    if args.threshold_steps < 2 or not (0 < args.threshold_min < args.threshold_max <= 1):
        raise ValueError("Threshold grid must satisfy 0 < min < max <= 1 with at least 2 steps")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    detection_names = [model.names[index] for index in sorted(model.names)]
    shared_names = ["background", *detection_names]
    name_to_index = {name: index for index, name in enumerate(shared_names)}
    images = image_paths(data_yaml, args.split)
    truth = load_truth(manifest, args.split)
    if not images:
        raise ValueError(f"No images found for split {args.split!r}")

    model.predict(source=str(images[0]), imgsz=args.imgsz, device=args.device, verbose=False)
    started = perf_counter()
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
    raw_scores: list[np.ndarray] = []
    y_true: list[int] = []
    groups: list[str] = []
    rows: list[dict[str, object]] = []
    for path, result in zip(images, results, strict=True):
        digest = sha256(path).lower()
        if digest not in truth:
            raise KeyError(f"Image absent from frozen manifest: {path.name}")
        class_scores = np.zeros(len(detection_names), dtype=float)
        if result.boxes is not None and len(result.boxes):
            confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
            for class_id, confidence in zip(class_ids, confidences):
                class_scores[class_id] = max(class_scores[class_id], confidence)
        target = truth[digest]["target"]
        group = truth[digest]["group"]
        raw_scores.append(class_scores)
        y_true.append(name_to_index[target])
        groups.append(group)
        row: dict[str, object] = {
            "path": str(path),
            "image_sha256": digest,
            "source_group_id": group,
            "true_class": target,
        }
        row.update(
            {
                f"max_confidence_{name}": float(class_scores[index])
                for index, name in enumerate(detection_names)
            }
        )
        rows.append(row)
    inference_seconds = perf_counter() - started

    y_array = np.asarray(y_true, dtype=int)
    score_array = np.vstack(raw_scores)
    group_array = np.asarray(groups)
    thresholds = np.linspace(
        args.threshold_min, args.threshold_max, args.threshold_steps, dtype=float
    )
    selected_threshold, curve = select_threshold(y_array, score_array, thresholds)
    selected_probabilities = shared_probabilities(score_array, selected_threshold)
    selected_metrics = metric_vector(y_array, selected_probabilities)

    with (output / "raw_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "threshold_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)

    report = {
        "schema_version": 1,
        "purpose": "validation_only_detector_operating_point_selection",
        "selection_rule": {
            "primary": "maximize_macro_f1",
            "secondary": "maximize_balanced_accuracy",
            "tie_breaker": "lowest_threshold",
            "grid": {
                "minimum": args.threshold_min,
                "maximum": args.threshold_max,
                "steps": args.threshold_steps,
            },
        },
        "selected_threshold": selected_threshold,
        "selected_split_metrics": selected_metrics,
        "selected_split_group_bootstrap_95_ci": group_intervals(
            y_array,
            selected_probabilities,
            group_array,
            args.bootstrap,
            args.seed,
        ),
        "model": {"path": str(model_path), "sha256": sha256(model_path)},
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
        "prediction_pass_wall_seconds": inference_seconds,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "test_usage": "Pass selected_threshold unchanged as --confidence to evaluate_yolo_detector.py on the frozen test split.",
        "probability_note": "The normalized shared scores reproduce the thresholded hard decision but are not calibrated probabilities.",
    }
    (output / "operating_point.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_threshold": selected_threshold,
                "selected_split_metrics": selected_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
