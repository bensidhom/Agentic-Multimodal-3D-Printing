"""Build plot-ready tables and publication figures for the vision task study."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_RUN = ROOT / "runs" / "classification" / "clinic_v2_yolo11n_cls_seed17"
DETECTOR_RUN = ROOT / "runs" / "detection" / "clinic_v2_yolo11n_detect_seed17"
METRICS_DIR = ROOT / "reports" / "metrics"
TABLES_DIR = ROOT / "reports" / "tables"
FIGURES_DIR = ROOT / "reports" / "figures"

NAVY = "#0B3C5D"
ORANGE = "#D35400"
TEAL = "#147D74"
GOLD = "#C69214"
GRAY = "#59636E"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_training_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def confusion(rows: list[dict[str, str]], names: list[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=int)
    for row in rows:
        matrix[lookup[row["true_class"]], lookup[row["predicted_class"]]] += 1
    return matrix


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#30343B",
            "axes.linewidth": 0.8,
            "grid.color": "#D8DDE3",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#1B1F23",
        va="top",
    )


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(FIGURES_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    figure.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def training_figure(classifier: dict[str, np.ndarray], detector: dict[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 7.5), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(classifier["epoch"], classifier["train/loss"], color=NAVY, lw=2.2, label="Train")
    axis.plot(classifier["epoch"], classifier["val/loss"], color=ORANGE, lw=2.2, label="Validation")
    axis.set(title="Classifier loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.grid(axis="y")
    axis.legend(ncol=2)
    panel_label(axis, "A")

    axis = axes[0, 1]
    axis.plot(
        classifier["epoch"],
        classifier["metrics/accuracy_top1"],
        color=TEAL,
        lw=2.4,
        marker="o",
        ms=3.2,
    )
    axis.set(title="Classifier validation accuracy", xlabel="Epoch", ylabel="Top-1 accuracy", ylim=(0.75, 1.015))
    axis.grid(axis="y")
    panel_label(axis, "B")

    axis = axes[1, 0]
    axis.plot(detector["epoch"], detector["metrics/mAP50(B)"], color=NAVY, lw=2.3, label="mAP@0.50")
    axis.plot(detector["epoch"], detector["metrics/mAP50-95(B)"], color=ORANGE, lw=2.3, label="mAP@0.50:0.95")
    best_index = int(np.argmax(0.1 * detector["metrics/mAP50(B)"] + 0.9 * detector["metrics/mAP50-95(B)"]))
    axis.scatter(
        detector["epoch"][best_index],
        detector["metrics/mAP50-95(B)"][best_index],
        s=58,
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=1.8,
        zorder=5,
        label=f"Retained epoch {int(detector['epoch'][best_index])}",
    )
    axis.set(title="Detector validation localization", xlabel="Epoch", ylabel="Mean average precision (mAP)", ylim=(0.2, 0.75))
    axis.grid(axis="y")
    axis.legend()
    panel_label(axis, "C")

    axis = axes[1, 1]
    axis.plot(detector["epoch"], detector["train/box_loss"], color=NAVY, lw=2.0, label="Train box")
    axis.plot(detector["epoch"], detector["val/box_loss"], color=NAVY, lw=2.0, ls="--", label="Validation box")
    axis.plot(detector["epoch"], detector["train/cls_loss"], color=ORANGE, lw=2.0, label="Train class")
    axis.plot(detector["epoch"], detector["val/cls_loss"], color=ORANGE, lw=2.0, ls="--", label="Validation class")
    axis.set(title="Detector optimization history", xlabel="Epoch", ylabel="Loss")
    axis.grid(axis="y")
    axis.legend(ncol=2, fontsize=9)
    panel_label(axis, "D")

    figure.suptitle("YOLO11n vision benchmark: complete training histories", fontsize=14, fontweight="bold")
    save_figure(figure, "vision_task_training_curves_seed17")


def comparison_figure(classifier: dict, detector: dict) -> None:
    cls_metrics = classifier["metrics"]
    det_metrics = detector["shared_image_level_metrics"]
    cls_ci = classifier["confidence_intervals"]
    det_ci = detector["shared_metric_confidence_intervals"]
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced\naccuracy"),
        ("macro_f1", "Macro F1"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)

    axis = axes[0]
    x = np.arange(len(metric_specs))
    width = 0.34
    cls_values = np.asarray([cls_metrics[key] for key, _ in metric_specs])
    det_values = np.asarray([det_metrics[key] for key, _ in metric_specs])
    cls_error = np.asarray(
        [[cls_values[i] - cls_ci[key]["lower_95"] for i, (key, _) in enumerate(metric_specs)],
         [cls_ci[key]["upper_95"] - cls_values[i] for i, (key, _) in enumerate(metric_specs)]]
    )
    det_error = np.asarray(
        [[det_values[i] - det_ci[key]["lower_95"] for i, (key, _) in enumerate(metric_specs)],
         [det_ci[key]["upper_95"] - det_values[i] for i, (key, _) in enumerate(metric_specs)]]
    )
    axis.errorbar(
        x - 0.08,
        cls_values,
        yerr=cls_error,
        fmt="o",
        ms=7,
        color=NAVY,
        elinewidth=1.7,
        capsize=4,
        label="Classification",
    )
    axis.errorbar(
        x + 0.08,
        det_values,
        yerr=det_error,
        fmt="o",
        ms=7,
        color=ORANGE,
        elinewidth=1.7,
        capsize=4,
        label="Detection",
    )
    for index, value in enumerate(cls_values):
        axis.text(index - 0.08, value + 0.004, f"{value:.3f}", ha="center", fontsize=8, color=NAVY)
    for index, value in enumerate(det_values):
        axis.text(index + 0.08, value + 0.004, f"{value:.3f}", ha="center", fontsize=8, color=ORANGE)
    axis.set(title="Shared image-level endpoint", ylabel="Score (95% group-bootstrap CI)", xticks=x, xticklabels=[label for _, label in metric_specs], ylim=(0.90, 1.012))
    axis.grid(axis="y")
    axis.legend(loc="lower left", fontsize=9)
    panel_label(axis, "A")

    axis = axes[1]
    latency_names = ["Median", "95th percentile"]
    cls_latency = [classifier["latency"]["median_ms"], classifier["latency"]["p95_ms"]]
    det_latency = [detector["latency"]["median_ms"], detector["latency"]["p95_ms"]]
    x = np.arange(2)
    cls_bars = axis.bar(x - width / 2, cls_latency, width, color=NAVY, label="Classification")
    det_bars = axis.bar(x + width / 2, det_latency, width, color=ORANGE, label="Detection")
    axis.bar_label(cls_bars, fmt="%.1f", padding=3, fontsize=8)
    axis.bar_label(det_bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set(title="Batch-one end-to-end latency", ylabel="Milliseconds", xticks=x, xticklabels=latency_names, ylim=(0, max(det_latency) * 1.22))
    axis.grid(axis="y")
    axis.text(
        0.02,
        0.96,
        f"Checkpoint size: {classifier['model']['size_mb']:.2f} vs {detector['model']['size_mb']:.2f} MB",
        transform=axis.transAxes,
        va="top",
        fontsize=8.5,
        color=GRAY,
    )
    panel_label(axis, "B")

    axis = axes[2]
    localization = detector["localization_metrics"]
    labels = ["Precision", "Recall", "AP@0.50", "AP@0.50:0.95"]
    values = [localization["precision"], localization["recall"], localization["ap50"], localization["ap50_95"]]
    bars = axis.bar(np.arange(4), values, color=[TEAL, TEAL, GOLD, GOLD])
    axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set(title="Detector localization endpoint", ylabel="Score", xticks=np.arange(4), xticklabels=labels, ylim=(0, 0.84))
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y")
    panel_label(axis, "C")

    figure.suptitle("Same-image classification versus object detection (frozen test split, n = 93)", fontsize=14, fontweight="bold")
    save_figure(figure, "vision_task_test_comparison_seed17")


def confusion_figure(classifier_rows: list[dict[str, str]], detector_rows: list[dict[str, str]], names: list[str]) -> None:
    matrices = [confusion(classifier_rows, names), confusion(detector_rows, names)]
    titles = ["Classification", "Detection as image decision"]
    colors = [NAVY, ORANGE]
    display_names = ["Background", "Spaghettification", "Underextrusion"]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
    for panel, (axis, matrix, title, color) in enumerate(zip(axes, matrices, titles, colors)):
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float), where=row_totals > 0)
        cmap = LinearSegmentedColormap.from_list("white_to_accent", ["#FFFFFF", color])
        axis.imshow(normalized, cmap=cmap, vmin=0, vmax=1, alpha=0.28)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]}\n({normalized[row, column] * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color="#1B1F23",
                    fontsize=10,
                    fontweight="bold" if row == column else "normal",
                )
        axis.set(
            title=title,
            xlabel="Predicted class",
            ylabel="True class",
            xticks=np.arange(len(names)),
            yticks=np.arange(len(names)),
            xticklabels=display_names,
            yticklabels=display_names,
        )
        axis.tick_params(length=0)
        axis.tick_params(axis="x", labelrotation=18)
        axis.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        axis.grid(which="minor", color="#AEB6BF", linewidth=0.8)
        axis.grid(which="major", visible=False)
        panel_label(axis, chr(ord("A") + panel))
    figure.suptitle("Frozen-test confusion matrices with row percentages", fontsize=14, fontweight="bold")
    save_figure(figure, "vision_task_confusion_matrices_seed17")


def write_outputs(classifier: dict, detector: dict, source_files: list[Path]) -> None:
    cls = classifier["metrics"]
    det = detector["shared_image_level_metrics"]
    loc = detector["localization_metrics"]
    rows = [
        {
            "task": "classification",
            "model": "YOLO11n-cls",
            "seed": 17,
            "test_images": classifier["dataset"]["images"],
            "accuracy": cls["accuracy"],
            "balanced_accuracy": cls["balanced_accuracy"],
            "macro_f1": cls["macro_f1"],
            "brier": cls["multiclass_brier"],
            "ece_15bin": cls["expected_calibration_error_15bin"],
            "localization_ap50": "",
            "localization_ap50_95": "",
            "median_latency_ms": classifier["latency"]["median_ms"],
            "p95_latency_ms": classifier["latency"]["p95_ms"],
            "checkpoint_mb": classifier["model"]["size_mb"],
        },
        {
            "task": "object_detection",
            "model": "YOLO11n",
            "seed": 17,
            "test_images": detector["dataset"]["images"],
            "accuracy": det["accuracy"],
            "balanced_accuracy": det["balanced_accuracy"],
            "macro_f1": det["macro_f1"],
            "brier": det["multiclass_brier"],
            "ece_15bin": det["expected_calibration_error_15bin"],
            "localization_ap50": loc["ap50"],
            "localization_ap50_95": loc["ap50_95"],
            "median_latency_ms": detector["latency"]["median_ms"],
            "p95_latency_ms": detector["latency"]["p95_ms"],
            "checkpoint_mb": detector["model"]["size_mb"],
        },
    ]
    csv_path = METRICS_DIR / "vision_task_comparison_seed17.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_index = [
        {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
        for path in source_files
    ]
    payload = {
        "schema_version": 1,
        "scope": "single-seed same-image benchmark; not a multi-seed final estimate",
        "selection": {
            "classifier": "best validation checkpoint after patience-based training",
            "detector": "best validation checkpoint; confidence 0.25 selected on validation only",
        },
        "rows": rows,
        "detector_per_class_localization": loc["per_class"],
        "source_files": source_index,
        "figure_files": sorted(str(path.resolve()) for path in FIGURES_DIR.glob("vision_task_*_seed17.*")),
    }
    json_path = METRICS_DIR / "vision_task_comparison_seed17.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    table = [
        "# Vision task comparison (seed 17)",
        "",
        "This is a same-image, single-seed benchmark. It is suitable as a candidate result and pipeline evidence, but final paper estimates require the registered additional seeds/models.",
        "",
        "| Task | Accuracy | Balanced accuracy | Macro F1 | AP50 | AP50-95 | Median latency | P95 latency | Model size |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Classification | {cls['accuracy']:.3f} | {cls['balanced_accuracy']:.3f} | {cls['macro_f1']:.3f} | — | — | {classifier['latency']['median_ms']:.1f} ms | {classifier['latency']['p95_ms']:.1f} ms | {classifier['model']['size_mb']:.2f} MB |",
        f"| Object detection | {det['accuracy']:.3f} | {det['balanced_accuracy']:.3f} | {det['macro_f1']:.3f} | {loc['ap50']:.3f} | {loc['ap50_95']:.3f} | {detector['latency']['median_ms']:.1f} ms | {detector['latency']['p95_ms']:.1f} ms | {detector['model']['size_mb']:.2f} MB |",
        "",
        "Detector image-level confidence was fixed at 0.25 on validation data. Detector decision scores are not calibrated probabilities; its Brier score/ECE are diagnostics and should not be presented as calibrated performance.",
        "",
        "All source logs, predictions, checkpoints, arguments and hashes are indexed in `../metrics/vision_task_comparison_seed17.json`.",
    ]
    (TABLES_DIR / "vision_task_comparison_seed17.md").write_text("\n".join(table) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (METRICS_DIR, TABLES_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    classifier_metrics_path = CLASSIFIER_RUN / "test_evaluation" / "metrics.json"
    detector_metrics_path = DETECTOR_RUN / "test_evaluation" / "metrics.json"
    classifier_predictions_path = CLASSIFIER_RUN / "test_evaluation" / "predictions.csv"
    detector_predictions_path = DETECTOR_RUN / "test_evaluation" / "predictions.csv"
    classifier_results_path = CLASSIFIER_RUN / "results.csv"
    detector_results_path = DETECTOR_RUN / "results.csv"
    operating_point_path = DETECTOR_RUN / "validation_operating_point" / "operating_point.json"
    source_files = [
        classifier_metrics_path,
        detector_metrics_path,
        classifier_predictions_path,
        detector_predictions_path,
        classifier_results_path,
        detector_results_path,
        operating_point_path,
        CLASSIFIER_RUN / "run_manifest.json",
        DETECTOR_RUN / "run_manifest.json",
    ]
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    classifier_metrics = json.loads(classifier_metrics_path.read_text(encoding="utf-8"))
    detector_metrics = json.loads(detector_metrics_path.read_text(encoding="utf-8"))
    classifier_training = read_training_csv(classifier_results_path)
    detector_training = read_training_csv(detector_results_path)
    classifier_predictions = read_predictions(classifier_predictions_path)
    detector_predictions = read_predictions(detector_predictions_path)
    names = classifier_metrics["dataset"]["classes"]

    training_figure(classifier_training, detector_training)
    comparison_figure(classifier_metrics, detector_metrics)
    confusion_figure(classifier_predictions, detector_predictions, names)
    write_outputs(classifier_metrics, detector_metrics, source_files)
    print(
        json.dumps(
            {
                "metrics_csv": str((METRICS_DIR / "vision_task_comparison_seed17.csv").resolve()),
                "metrics_json": str((METRICS_DIR / "vision_task_comparison_seed17.json").resolve()),
                "table": str((TABLES_DIR / "vision_task_comparison_seed17.md").resolve()),
                "figures": sorted(path.name for path in FIGURES_DIR.glob("vision_task_*_seed17.*")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
