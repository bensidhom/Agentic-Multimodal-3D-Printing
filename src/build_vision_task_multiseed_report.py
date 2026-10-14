"""Compare purged-view classification candidates with nano detectors."""

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
METRICS_DIR = ROOT / "reports/metrics"
TABLES_DIR = ROOT / "reports/tables"
FIGURES_DIR = ROOT / "reports/figures"
SEEDS = (17, 29, 43)
INK = "#20252A"

COLORS = {
    "YOLO11n-cls": "#153A59",
    "ResNet-18": "#788592",
    "MobileNetV3-Small": "#147D74",
    "EfficientNet-B0": "#C85B21",
    "YOLO11n detector": "#C69214",
    "YOLO26n detector": "#7B4F9D",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#313840",
            "axes.linewidth": 0.8,
            "grid.color": "#D9DEE3",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "legend.frameon": False,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.12, 1.08, label, transform=axis.transAxes, fontsize=12, fontweight="bold", color=INK, va="top")


def save(figure: plt.Figure, stem: str) -> None:
    figure.savefig(FIGURES_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def collect() -> tuple[list[dict], list[Path]]:
    classifier_rows_path = METRICS_DIR / "vision_classifier_multiseed_rows.csv"
    if not classifier_rows_path.is_file():
        raise FileNotFoundError(classifier_rows_path)
    with classifier_rows_path.open(newline="", encoding="utf-8-sig") as stream:
        classifier_rows = list(csv.DictReader(stream))
    rows: list[dict] = []
    for row in classifier_rows:
        rows.append(
            {
                "model": row["model"],
                "task": "classification",
                "seed": int(row["seed"]),
                "accuracy": float(row["accuracy"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "macro_f1": float(row["macro_f1"]),
                "ap50": "",
                "ap50_95": "",
                "median_latency_ms": float(row["median_latency_ms"]),
                "p95_latency_ms": float(row["p95_latency_ms"]),
                "checkpoint_mb": float(row["checkpoint_mb"]),
                "selected_threshold": "",
            }
        )

    sources: list[Path] = [classifier_rows_path, METRICS_DIR / "vision_classifier_multiseed.json"]
    for detector_model, run_prefix in (
        ("YOLO11n detector", "clinic_v2_purged_yolo11n_detect_seed"),
        ("YOLO26n detector", "clinic_v2_purged_yolo26n_detect_seed"),
    ):
        for seed in SEEDS:
            run = ROOT / f"runs/detection/{run_prefix}{seed}"
            metrics_path = run / "test_evaluation/metrics.json"
            operating_path = run / "validation_operating_point/operating_point.json"
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            operating = json.loads(operating_path.read_text(encoding="utf-8"))
            shared = payload["shared_image_level_metrics"]
            localization = payload["localization_metrics"]
            rows.append(
                {
                    "model": detector_model,
                    "task": "object_detection",
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
            )
            sources.extend([metrics_path, metrics_path.with_name("predictions.csv"), operating_path, run / "results.csv", run / "run_manifest.json"])
    return rows, sources


def summarize(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for model in COLORS:
        subset = [row for row in rows if row["model"] == model]
        summary: dict[str, object] = {"model": model, "task": subset[0]["task"], "seeds": len(subset)}
        keys = ["accuracy", "balanced_accuracy", "macro_f1", "median_latency_ms", "p95_latency_ms", "checkpoint_mb"]
        if subset[0]["task"] == "object_detection":
            keys += ["ap50", "ap50_95", "selected_threshold"]
        for key in keys:
            values = np.asarray([float(row[key]) for row in subset], dtype=float)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_sd"] = float(values.std(ddof=1))
            summary[f"{key}_min"] = float(values.min())
            summary[f"{key}_max"] = float(values.max())
        output.append(summary)
    return output


def make_figure(rows: list[dict], summaries: list[dict]) -> None:
    models = list(COLORS)
    x = np.arange(len(models))
    jitter = {17: -0.09, 29: 0.0, 43: 0.09}
    figure, axes = plt.subplots(2, 2, figsize=(12.3, 8.1), constrained_layout=True)

    axis = axes[0, 0]
    for index, model in enumerate(models):
        subset = [row for row in rows if row["model"] == model]
        for row in subset:
            axis.scatter(index + jitter[row["seed"]], row["macro_f1"], s=42, color=COLORS[model], alpha=0.84, edgecolor="white", linewidth=0.7)
            axis.text(index + jitter[row["seed"]], row["macro_f1"] + 0.006, str(row["seed"]), ha="center", fontsize=7.2, color=COLORS[model])
        summary = next(item for item in summaries if item["model"] == model)
        axis.errorbar(index, summary["macro_f1_mean"], yerr=summary["macro_f1_sd"], fmt="D", ms=5.5, color=INK, mfc="white", capsize=4, lw=1.4, zorder=5)
    axis.axvline(3.5, color="#AEB6BF", ls="--", lw=1.0)
    axis.text(3.25, 0.79, "classifiers", ha="right", fontsize=8, color="#68737D")
    axis.text(3.75, 0.79, "detectors", ha="left", fontsize=8, color="#68737D")
    axis.set(title="Shared endpoint: macro F1 by seed", ylabel="Test macro F1", xticks=x, xticklabels=models, ylim=(0.78, 1.02))
    axis.tick_params(axis="x", labelrotation=18)
    axis.grid(axis="y")
    panel_label(axis, "A")

    axis = axes[0, 1]
    metric_keys = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced\naccuracy"), ("macro_f1", "Macro F1")]
    width = 0.16
    offsets = np.linspace(-0.32, 0.32, len(models))
    for model, offset in zip(models, offsets):
        summary = next(item for item in summaries if item["model"] == model)
        means = [summary[f"{key}_mean"] for key, _ in metric_keys]
        sds = [summary[f"{key}_sd"] for key, _ in metric_keys]
        axis.bar(np.arange(3) + offset, means, width, yerr=sds, color=COLORS[model], alpha=0.72, edgecolor=COLORS[model], linewidth=0.7, capsize=2.5, label=model)
    axis.set(title="Shared image-decision metrics", ylabel="Mean score ± SD", xticks=np.arange(3), xticklabels=[label for _, label in metric_keys], ylim=(0.76, 1.02))
    axis.grid(axis="y")
    axis.legend(loc="lower left", fontsize=7.2, ncol=2)
    panel_label(axis, "B")

    axis = axes[1, 0]
    label_offsets = {
        "EfficientNet-B0": (7, -13),
        "YOLO11n detector": (7, 10),
        "YOLO26n detector": (7, -13),
    }
    for model in models:
        summary = next(item for item in summaries if item["model"] == model)
        axis.errorbar(summary["median_latency_ms_mean"], summary["macro_f1_mean"], xerr=summary["median_latency_ms_sd"], yerr=summary["macro_f1_sd"], fmt="o", ms=8, color=COLORS[model], elinewidth=1.3, capsize=3)
        axis.annotate(
            model,
            (summary["median_latency_ms_mean"], summary["macro_f1_mean"]),
            xytext=label_offsets.get(model, (6, 4)),
            textcoords="offset points",
            fontsize=8,
            color=COLORS[model],
        )
    axis.set(title="Shared performance–latency trade-off", xlabel="Median batch-one end-to-end latency (ms)", ylabel="Mean test macro F1", xlim=(0, None), ylim=(0.80, 0.99))
    axis.grid()
    panel_label(axis, "C")

    axis = axes[1, 1]
    detector_rows = [row for row in rows if row["task"] == "object_detection"]
    detector_models = ["YOLO11n detector", "YOLO26n detector"]
    local_x = np.arange(2)
    local_labels = ["AP@0.50", "AP@0.50:0.95"]
    local_keys = ["ap50", "ap50_95"]
    architecture_offsets = {"YOLO11n detector": -0.18, "YOLO26n detector": 0.18}
    seed_offsets = {17: -0.045, 29: 0.0, 43: 0.045}
    for detector_model in detector_models:
        for seed in SEEDS:
            row = next(item for item in detector_rows if item["model"] == detector_model and item["seed"] == seed)
            values = [row[key] for key in local_keys]
            offset = architecture_offsets[detector_model] + seed_offsets[seed]
            axis.scatter(local_x + offset, values, s=44, color=COLORS[detector_model], alpha=0.76, edgecolor="white", linewidth=0.8)
            for index, value in enumerate(values):
                axis.text(index + offset, value + 0.012, str(seed), ha="center", fontsize=6.8, color=COLORS[detector_model])
        detector_summary = next(item for item in summaries if item["model"] == detector_model)
        means = [detector_summary[f"{key}_mean"] for key in local_keys]
        sds = [detector_summary[f"{key}_sd"] for key in local_keys]
        axis.errorbar(local_x + architecture_offsets[detector_model], means, yerr=sds, fmt="D", ms=6, color=COLORS[detector_model], mfc="white", capsize=5, lw=1.5, zorder=5, label=detector_model)
    axis.set(title="Detector-only localization endpoint", ylabel="Mean average precision", xticks=local_x, xticklabels=local_labels, ylim=(0, 0.82))
    axis.grid(axis="y")
    axis.legend(loc="upper right", fontsize=7.5)
    axis.text(0.02, 0.04, "Circles = seeds; diamond = mean ± SD", transform=axis.transAxes, fontsize=8, color="#68737D")
    panel_label(axis, "D")

    figure.suptitle("Classification versus object detection on the purged Clinic-v2 view (three seeds)", fontsize=14, fontweight="bold", color=INK)
    save(figure, "vision_task_multiseed")


def write_outputs(rows: list[dict], summaries: list[dict], sources: list[Path]) -> None:
    rows_path = METRICS_DIR / "vision_task_multiseed_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = METRICS_DIR / "vision_task_multiseed_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        summary_fields = list(summaries[0]) + sorted(
            set().union(*(item.keys() for item in summaries)) - set(summaries[0])
        )
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "schema_version": 1,
        "purpose": "three-seed classification-versus-detection comparison on the purged chronological sensitivity view",
        "interpretation_guard": "Only three acquisition sequences exist; this is not independent-print validation.",
        "detector_score_guard": "Detector image-decision scores are threshold-normalized and uncalibrated; Brier/ECE are intentionally excluded from the cross-task comparison.",
        "rows": rows,
        "summaries": summaries,
        "source_files": [{"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size} for path in sources],
        "figures": sorted(str(path.resolve()) for path in FIGURES_DIR.glob("vision_task_multiseed.*")),
    }
    (METRICS_DIR / "vision_task_multiseed.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Classification versus object detection: three-seed purged-view comparison",
        "",
        "Shared image-level values are mean ± sample SD across seeds 17, 29, and 43 on the same 219-image test set. This is a temporal-overlap sensitivity analysis, not independent-print validation.",
        "",
        "| Model | Task | Accuracy | Balanced accuracy | Macro F1 | AP50 | AP50:95 | Median latency | Size |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(summaries, key=lambda row: row["macro_f1_mean"], reverse=True):
        ap50 = "—" if item["task"] == "classification" else f"{item['ap50_mean']:.3f} ± {item['ap50_sd']:.3f}"
        ap95 = "—" if item["task"] == "classification" else f"{item['ap50_95_mean']:.3f} ± {item['ap50_95_sd']:.3f}"
        lines.append(
            f"| {item['model']} | {item['task'].replace('_', ' ')} | {item['accuracy_mean']:.3f} ± {item['accuracy_sd']:.3f} | "
            f"{item['balanced_accuracy_mean']:.3f} ± {item['balanced_accuracy_sd']:.3f} | {item['macro_f1_mean']:.3f} ± {item['macro_f1_sd']:.3f} | "
            f"{ap50} | {ap95} | {item['median_latency_ms_mean']:.1f} ± {item['median_latency_ms_sd']:.1f} ms | {item['checkpoint_mb_mean']:.1f} MB |"
        )
    best_detector = max(
        (item for item in summaries if item["task"] == "object_detection"),
        key=lambda item: item["macro_f1_mean"],
    )
    lines.extend(
        [
            "",
            f"{best_detector['model']} is the strongest detector by mean shared-endpoint macro F1 and supplies localization. All detector thresholds were selected separately on validation data; localization AP and image-level macro F1 are retained as distinct endpoints.",
            "",
            "Detector Brier score and ECE are not included because its threshold-normalized image-decision scores are not calibrated probabilities.",
        ]
    )
    (TABLES_DIR / "vision_task_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (METRICS_DIR, TABLES_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    configure()
    rows, sources = collect()
    summaries = summarize(rows)
    make_figure(rows, summaries)
    write_outputs(rows, summaries, sources)
    print(json.dumps({"rows": len(rows), "models": len(summaries), "report": str((METRICS_DIR / "vision_task_multiseed.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
