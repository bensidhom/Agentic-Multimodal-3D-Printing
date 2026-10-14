"""Create auditable multi-seed classifier tables and publication figures."""

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

MODELS = {
    "YOLO11n-cls": "clinic_v2_purged_yolo11n_cls_seed{seed}",
    "ResNet-18": "clinic_v2_purged_resnet18_seed{seed}",
    "MobileNetV3-Small": "clinic_v2_purged_mobilenet_v3_small_seed{seed}",
    "EfficientNet-B0": "clinic_v2_purged_efficientnet_b0_seed{seed}",
}
COLORS = {
    "YOLO11n-cls": "#153A59",
    "ResNet-18": "#6B7785",
    "MobileNetV3-Small": "#147D74",
    "EfficientNet-B0": "#C85B21",
}
INK = "#20252A"


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
            "font.size": 9.5,
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
    rows: list[dict] = []
    sources: list[Path] = []
    for model, pattern in MODELS.items():
        for seed in SEEDS:
            path = ROOT / "runs/classification" / pattern.format(seed=seed) / "test_evaluation/metrics.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            model_data = payload["model"]
            size = model_data.get("size_mb", model_data.get("checkpoint_size_mb"))
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "test_images": payload["dataset"]["images"],
                    "source_groups": payload["dataset"]["source_groups"],
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "brier": metrics["multiclass_brier"],
                    "ece_15bin": metrics["expected_calibration_error_15bin"],
                    "median_latency_ms": payload["latency"]["median_ms"],
                    "p95_latency_ms": payload["latency"]["p95_ms"],
                    "checkpoint_mb": size,
                    "metrics_path": str(path.resolve()),
                }
            )
            sources.extend([path, path.with_name("predictions.csv")])
    return rows, sources


def summarize(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "brier", "ece_15bin", "median_latency_ms", "p95_latency_ms", "checkpoint_mb")
    for model in MODELS:
        subset = [row for row in rows if row["model"] == model]
        summary: dict[str, object] = {"model": model, "seeds": len(subset)}
        for key in keys:
            values = np.asarray([row[key] for row in subset], dtype=float)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_sd"] = float(values.std(ddof=1))
            summary[f"{key}_min"] = float(values.min())
            summary[f"{key}_max"] = float(values.max())
        output.append(summary)
    return output


def figure(rows: list[dict], summaries: list[dict]) -> None:
    models = list(MODELS)
    x = np.arange(len(models))
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.1), constrained_layout=True)

    axis = axes[0, 0]
    jitter = {17: -0.09, 29: 0.0, 43: 0.09}
    for index, model in enumerate(models):
        subset = [row for row in rows if row["model"] == model]
        for row in subset:
            axis.scatter(index + jitter[row["seed"]], row["macro_f1"], s=43, color=COLORS[model], alpha=0.82, edgecolor="white", linewidth=0.7)
            axis.text(index + jitter[row["seed"]], row["macro_f1"] + 0.006, str(row["seed"]), ha="center", fontsize=7.3, color=COLORS[model])
        mean = next(item["macro_f1_mean"] for item in summaries if item["model"] == model)
        sd = next(item["macro_f1_sd"] for item in summaries if item["model"] == model)
        axis.errorbar(index, mean, yerr=sd, fmt="D", ms=5.5, color=INK, mfc="white", capsize=4, lw=1.4, zorder=5)
    axis.set(title="Macro F1 by random seed", ylabel="Test macro F1", xticks=x, xticklabels=models, ylim=(0.78, 1.02))
    axis.tick_params(axis="x", labelrotation=18)
    axis.grid(axis="y")
    axis.text(0.02, 0.04, "Colored circles = seeds; diamond = mean ± SD", transform=axis.transAxes, fontsize=8, color="#66717C")
    panel_label(axis, "A")

    axis = axes[0, 1]
    metrics = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced\naccuracy"), ("macro_f1", "Macro F1")]
    width = 0.19
    offsets = np.linspace(-0.285, 0.285, len(models))
    for model, offset in zip(models, offsets):
        summary = next(item for item in summaries if item["model"] == model)
        means = [summary[f"{key}_mean"] for key, _ in metrics]
        sds = [summary[f"{key}_sd"] for key, _ in metrics]
        axis.bar(
            np.arange(3) + offset,
            means,
            width,
            yerr=sds,
            color=COLORS[model],
            alpha=0.72,
            edgecolor=COLORS[model],
            linewidth=0.7,
            capsize=2.5,
            label=model,
        )
    axis.set(title="Three-seed performance summary", ylabel="Mean score ± SD", xticks=np.arange(3), xticklabels=[label for _, label in metrics], ylim=(0.76, 1.02))
    axis.grid(axis="y")
    axis.legend(loc="lower left", fontsize=7.8, ncol=2)
    panel_label(axis, "B")

    axis = axes[1, 0]
    for model in models:
        summary = next(item for item in summaries if item["model"] == model)
        axis.errorbar(
            summary["median_latency_ms_mean"],
            summary["macro_f1_mean"],
            xerr=summary["median_latency_ms_sd"],
            yerr=summary["macro_f1_sd"],
            fmt="o",
            ms=8,
            color=COLORS[model],
            elinewidth=1.3,
            capsize=3,
        )
        axis.annotate(model, (summary["median_latency_ms_mean"], summary["macro_f1_mean"]), xytext=(6, 5), textcoords="offset points", fontsize=8.3, color=COLORS[model])
    axis.set(title="Accuracy–latency trade-off", xlabel="Median batch-one end-to-end latency (ms)", ylabel="Mean test macro F1", xlim=(0, None), ylim=(0.80, 0.99))
    axis.grid()
    panel_label(axis, "C")

    axis = axes[1, 1]
    width = 0.35
    brier = [item["brier_mean"] for item in summaries]
    ece = [item["ece_15bin_mean"] for item in summaries]
    brier_sd = [item["brier_sd"] for item in summaries]
    ece_sd = [item["ece_15bin_sd"] for item in summaries]
    first = axis.bar(x - width / 2, brier, width, yerr=brier_sd, color="white", edgecolor=[COLORS[m] for m in models], hatch="///", linewidth=1.4, capsize=2.5, label="Multiclass Brier")
    second = axis.bar(x + width / 2, ece, width, yerr=ece_sd, color=[COLORS[m] for m in models], alpha=0.68, linewidth=0.6, capsize=2.5, label="ECE (15 bins)")
    axis.set(title="Confidence diagnostics", ylabel="Lower is better (mean ± SD)", xticks=x, xticklabels=models, ylim=(0, max(max(np.asarray(brier) + np.asarray(brier_sd)), max(np.asarray(ece) + np.asarray(ece_sd))) * 1.32))
    axis.tick_params(axis="x", labelrotation=18)
    axis.grid(axis="y")
    axis.legend(fontsize=8)
    panel_label(axis, "D")

    figure.suptitle("Clinic-v2 purged sensitivity benchmark: classifier architecture comparison (n = 219)", fontsize=14, fontweight="bold", color=INK)
    save(figure, "vision_classifier_multiseed")


def write_outputs(rows: list[dict], summaries: list[dict], sources: list[Path]) -> None:
    row_csv = METRICS_DIR / "vision_classifier_multiseed_rows.csv"
    with row_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_csv = METRICS_DIR / "vision_classifier_multiseed_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    payload = {
        "schema_version": 1,
        "purpose": "three-seed classifier architecture comparison on purged chronological sensitivity view",
        "interpretation_guard": "Only three acquisition sequences exist; this is not independent-print validation.",
        "seeds": list(SEEDS),
        "rows": rows,
        "summaries": summaries,
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sources
        ],
        "figures": sorted(str(path.resolve()) for path in FIGURES_DIR.glob("vision_classifier_multiseed.*")),
    }
    (METRICS_DIR / "vision_classifier_multiseed.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Classifier architecture comparison: purged Clinic-v2 view",
        "",
        "Values are mean ± sample SD across seeds 17, 29, and 43 on the same 219-image test set. This is a temporal-overlap sensitivity analysis, not independent-print validation.",
        "",
        "| Model | Accuracy | Balanced accuracy | Macro F1 | Brier | ECE | Median latency | Checkpoint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(summaries, key=lambda row: row["macro_f1_mean"], reverse=True):
        lines.append(
            f"| {item['model']} | {item['accuracy_mean']:.3f} ± {item['accuracy_sd']:.3f} | "
            f"{item['balanced_accuracy_mean']:.3f} ± {item['balanced_accuracy_sd']:.3f} | "
            f"{item['macro_f1_mean']:.3f} ± {item['macro_f1_sd']:.3f} | "
            f"{item['brier_mean']:.3f} ± {item['brier_sd']:.3f} | "
            f"{item['ece_15bin_mean']:.3f} ± {item['ece_15bin_sd']:.3f} | "
            f"{item['median_latency_ms_mean']:.1f} ± {item['median_latency_ms_sd']:.1f} ms | "
            f"{item['checkpoint_mb_mean']:.1f} MB |"
        )
    lines.extend(
        [
            "",
            "Selection conclusion: EfficientNet-B0 has the highest mean macro F1. YOLO11n-cls is smaller and faster, while ResNet-18 is fastest but much larger. Report all seed-level values because optimization variance is material.",
        ]
    )
    (TABLES_DIR / "vision_classifier_multiseed.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (METRICS_DIR, TABLES_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    configure()
    rows, sources = collect()
    summaries = summarize(rows)
    figure(rows, summaries)
    write_outputs(rows, summaries, sources)
    print(json.dumps({"rows": len(rows), "models": len(summaries), "output": str((METRICS_DIR / "vision_classifier_multiseed.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
