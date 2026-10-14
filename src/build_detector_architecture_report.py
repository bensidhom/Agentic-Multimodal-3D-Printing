"""Build the three-seed YOLO11n-versus-YOLO26n detector report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports/metrics"
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures"
SEEDS = (17, 29, 43)
MODELS = {
    "YOLO11n": ("clinic_v2_purged_yolo11n_detect_seed", "#153A59"),
    "YOLO26n": ("clinic_v2_purged_yolo26n_detect_seed", "#C85B21"),
}
CLASSES = ("spaghettification", "underextrusion")
INK = "#20252A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect() -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    sources: list[dict] = []
    for model, (prefix, _) in MODELS.items():
        for seed in SEEDS:
            run = ROOT / "runs/detection" / f"{prefix}{seed}"
            metrics_path = run / "test_evaluation/metrics.json"
            operating_path = run / "validation_operating_point/operating_point.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            operating = json.loads(operating_path.read_text(encoding="utf-8"))
            shared = payload["shared_image_level_metrics"]
            localization = payload["localization_metrics"]
            row = {
                "model": model,
                "seed": seed,
                "accuracy": shared["accuracy"],
                "balanced_accuracy": shared["balanced_accuracy"],
                "macro_f1": shared["macro_f1"],
                "ap50": localization["ap50"],
                "ap50_95": localization["ap50_95"],
                "median_latency_ms": payload["latency"]["median_ms"],
                "p95_latency_ms": payload["latency"]["p95_ms"],
                "checkpoint_mb": payload["model"]["size_mb"],
                "selected_threshold": operating["selected_threshold"],
            }
            for class_name in CLASSES:
                per_class = localization["per_class"][class_name]
                for key in ("precision", "recall", "ap50", "ap50_95"):
                    row[f"{class_name}_{key}"] = per_class[key]
            rows.append(row)
            for path in (metrics_path, metrics_path.with_name("predictions.csv"), operating_path, run / "results.csv", run / "run_manifest.json"):
                sources.append({"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size})
    return rows, sources


def summarize(rows: list[dict]) -> list[dict]:
    keys = [
        "accuracy", "balanced_accuracy", "macro_f1", "ap50", "ap50_95",
        "median_latency_ms", "p95_latency_ms", "checkpoint_mb", "selected_threshold",
    ] + [f"{class_name}_{key}" for class_name in CLASSES for key in ("precision", "recall", "ap50", "ap50_95")]
    output = []
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        summary = {"model": model, "seeds": len(subset)}
        for key in keys:
            values = np.asarray([row[key] for row in subset], dtype=float)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_sd"] = float(values.std(ddof=1))
            summary[f"{key}_min"] = float(values.min())
            summary[f"{key}_max"] = float(values.max())
        output.append(summary)
    return output


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.12, 1.08, label, transform=axis.transAxes, fontsize=12, fontweight="bold", color=INK, va="top")


def make_figure(rows: list[dict], summaries: list[dict]) -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9.3, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.edgecolor": "#313840", "grid.color": "#D9DEE3",
        "grid.alpha": 0.85, "legend.frameon": False,
    })
    figure, axes = plt.subplots(2, 2, figsize=(12.1, 8.0), constrained_layout=True)
    model_names = list(MODELS)
    colors = {name: values[1] for name, values in MODELS.items()}
    x = np.arange(len(model_names))
    jitter = {17: -0.08, 29: 0.0, 43: 0.08}

    axis = axes[0, 0]
    for index, model in enumerate(model_names):
        subset = [row for row in rows if row["model"] == model]
        for row in subset:
            axis.scatter(index + jitter[row["seed"]], row["macro_f1"], s=48, color=colors[model], edgecolor="white", linewidth=0.7)
            axis.text(index + jitter[row["seed"]], row["macro_f1"] + 0.008, str(row["seed"]), ha="center", fontsize=7.5, color=colors[model])
        summary = next(item for item in summaries if item["model"] == model)
        axis.errorbar(index, summary["macro_f1_mean"], yerr=summary["macro_f1_sd"], fmt="D", ms=6, color=INK, mfc="white", capsize=5, lw=1.5)
    axis.set(title="Image-level decision endpoint", ylabel="Test macro F1", xticks=x, xticklabels=model_names, ylim=(0.78, 1.02))
    axis.grid(axis="y")
    panel_label(axis, "A")

    axis = axes[0, 1]
    metric_keys = (("ap50", "AP@0.50"), ("ap50_95", "AP@0.50:0.95"))
    width = 0.34
    for index, model in enumerate(model_names):
        summary = next(item for item in summaries if item["model"] == model)
        means = [summary[f"{key}_mean"] for key, _ in metric_keys]
        sds = [summary[f"{key}_sd"] for key, _ in metric_keys]
        axis.bar(np.arange(2) + (index - 0.5) * width, means, width, yerr=sds, capsize=3, color=colors[model], alpha=0.78, label=model)
    axis.set(title="Localization performance", ylabel="Mean average precision ± SD", xticks=np.arange(2), xticklabels=[label for _, label in metric_keys], ylim=(0, 0.82))
    axis.grid(axis="y")
    axis.legend(fontsize=8)
    panel_label(axis, "B")

    axis = axes[1, 0]
    locations = np.arange(len(CLASSES))
    subwidth = 0.18
    offsets = (-0.27, -0.09, 0.09, 0.27)
    series = []
    for model in model_names:
        summary = next(item for item in summaries if item["model"] == model)
        series.extend([
            (model, "AP@0.50", [summary[f"{class_name}_ap50_mean"] for class_name in CLASSES], colors[model], 0.82),
            (model, "AP@0.50:0.95", [summary[f"{class_name}_ap50_95_mean"] for class_name in CLASSES], colors[model], 0.42),
        ])
    for offset, (model, metric, values, color, alpha) in zip(offsets, series):
        axis.bar(locations + offset, values, subwidth, color=color, alpha=alpha, label=f"{model} {metric}")
    axis.set(title="Per-class localization", ylabel="Mean average precision", xticks=locations, xticklabels=[name.replace("underextrusion", "under-extrusion") for name in CLASSES], ylim=(0, 1.0))
    axis.grid(axis="y")
    axis.legend(fontsize=7.4, ncol=2, loc="upper left")
    panel_label(axis, "C")

    axis = axes[1, 1]
    label_offsets = {"YOLO11n": (7, 8), "YOLO26n": (7, -14)}
    for model in model_names:
        summary = next(item for item in summaries if item["model"] == model)
        axis.errorbar(summary["median_latency_ms_mean"], summary["ap50_mean"], xerr=summary["median_latency_ms_sd"], yerr=summary["ap50_sd"], fmt="o", ms=9, color=colors[model], capsize=4, elinewidth=1.4)
        axis.annotate(model, (summary["median_latency_ms_mean"], summary["ap50_mean"]), xytext=label_offsets[model], textcoords="offset points", fontsize=9, color=colors[model])
    axis.set(title="Localization–latency trade-off", xlabel="Median batch-one end-to-end latency (ms)", ylabel="Mean AP@0.50", xlim=(0, None), ylim=(0.55, 0.80))
    axis.grid()
    panel_label(axis, "D")

    figure.suptitle("Nano-detector architecture comparison on the purged Clinic-v2 view", fontsize=14, fontweight="bold", color=INK)
    figure.savefig(FIGURES / "vision_detector_architectures.png", dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURES / "vision_detector_architectures.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_outputs(rows: list[dict], summaries: list[dict], sources: list[dict]) -> None:
    with (METRICS / "vision_detector_architectures_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary_fields = sorted(set().union(*(item.keys() for item in summaries)))
    with (METRICS / "vision_detector_architectures_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader(); writer.writerows(summaries)
    payload = {
        "schema_version": 1,
        "purpose": "three-seed YOLO11n versus YOLO26n detector comparison on the purged chronological sensitivity view",
        "interpretation_guard": "Only three acquisition sequences exist; this is not independent-print validation.",
        "rows": rows,
        "summaries": summaries,
        "source_files": sources,
        "figures": [str((FIGURES / "vision_detector_architectures.png").resolve()), str((FIGURES / "vision_detector_architectures.pdf").resolve())],
    }
    (METRICS / "vision_detector_architectures.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Nano-detector architecture comparison", "",
        "Values are mean ± sample SD across seeds 17, 29, and 43 on the same 219-image purged-view test set. This is an archived temporal-overlap sensitivity analysis, not independent-print validation.", "",
        "| Detector | Accuracy | Balanced accuracy | Macro F1 | AP50 | AP50:95 | Median latency | Size | Threshold range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in sorted(summaries, key=lambda item: item["macro_f1_mean"], reverse=True):
        lines.append(
            f"| {summary['model']} | {summary['accuracy_mean']:.3f} ± {summary['accuracy_sd']:.3f} | {summary['balanced_accuracy_mean']:.3f} ± {summary['balanced_accuracy_sd']:.3f} | "
            f"{summary['macro_f1_mean']:.3f} ± {summary['macro_f1_sd']:.3f} | {summary['ap50_mean']:.3f} ± {summary['ap50_sd']:.3f} | {summary['ap50_95_mean']:.3f} ± {summary['ap50_95_sd']:.3f} | "
            f"{summary['median_latency_ms_mean']:.1f} ± {summary['median_latency_ms_sd']:.1f} ms | {summary['checkpoint_mb_mean']:.1f} MB | "
            f"{summary['selected_threshold_min']:.2f}–{summary['selected_threshold_max']:.2f} |"
        )
    lines += ["", "Thresholds were selected on validation data separately for every fitted seed. Image-level macro F1 and localization AP answer different questions and are retained together."]
    (TABLES / "vision_detector_architectures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (METRICS, TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    rows, sources = collect()
    summaries = summarize(rows)
    make_figure(rows, summaries)
    write_outputs(rows, summaries, sources)
    print(json.dumps({"rows": len(rows), "models": len(summaries), "report": str((METRICS / "vision_detector_architectures.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
