"""Build publication artifacts comparing original and purged Clinic-v2 splits."""

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
REPORTS = ROOT / "reports"
METRICS_DIR = REPORTS / "metrics"
TABLES_DIR = REPORTS / "tables"
FIGURES_DIR = REPORTS / "figures"

ORIGINAL_CLASSIFIER = ROOT / "runs/classification/clinic_v2_yolo11n_cls_seed17/test_evaluation"
ORIGINAL_DETECTOR = ROOT / "runs/detection/clinic_v2_yolo11n_detect_seed17/test_evaluation"
PURGED_CLASSIFIER = ROOT / "runs/classification/clinic_v2_purged_yolo11n_cls_seed17/test_evaluation"
PURGED_DETECTOR = ROOT / "runs/detection/clinic_v2_purged_yolo11n_detect_seed17/test_evaluation"

NAVY = "#153A59"
BLUE = "#3C78A8"
ORANGE = "#C85B21"
AMBER = "#D59B2D"
TEAL = "#147D74"
INK = "#20252A"
GRAY = "#67727E"
PALE = "#EEF2F5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


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
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#313840",
            "axes.linewidth": 0.8,
            "grid.color": "#D9DEE3",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
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
        color=INK,
        va="top",
    )


def save(figure: plt.Figure, stem: str) -> None:
    figure.savefig(FIGURES_DIR / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def image_metrics(payload: dict, task: str) -> tuple[dict, dict]:
    if task == "classification":
        return payload["metrics"], payload["confidence_intervals"]
    return payload["shared_image_level_metrics"], payload["shared_metric_confidence_intervals"]


def metric_figure(
    original_cls: dict,
    original_det: dict,
    purged_cls: dict,
    purged_det: dict,
    original_audit: dict,
    purged_audit: dict,
) -> None:
    payloads = [original_cls, original_det, purged_cls, purged_det]
    task_names = ["classification", "detection", "classification", "detection"]
    labels = ["Original\nclassification", "Original\ndetection", "Purged\nclassification", "Purged\ndetection"]
    colors = [NAVY, ORANGE, BLUE, AMBER]
    hatches = ["///", "///", "", ""]
    metrics = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("macro_f1", "Macro F1")]

    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.1), constrained_layout=True)

    axis = axes[0, 0]
    x = np.arange(len(metrics))
    offsets = np.linspace(-0.27, 0.27, 4)
    for payload, task, label, color, offset in zip(payloads, task_names, labels, colors, offsets):
        values, intervals = image_metrics(payload, task)
        estimates = np.asarray([values[key] for key, _ in metrics])
        lower = np.asarray([intervals[key]["lower_95"] for key, _ in metrics])
        upper = np.asarray([intervals[key]["upper_95"] for key, _ in metrics])
        axis.errorbar(
            x + offset,
            estimates,
            yerr=np.vstack([estimates - lower, upper - estimates]),
            fmt="o",
            ms=5.7,
            color=color,
            elinewidth=1.4,
            capsize=2.8,
            label=label.replace("\n", " "),
        )
    axis.axhspan(0.0, 0.80, color=PALE, alpha=0.45, zorder=-5)
    axis.set(
        title="Shared image-level performance",
        ylabel="Score (95% source-group bootstrap CI)",
        xticks=x,
        xticklabels=[label for _, label in metrics],
        ylim=(0.64, 1.025),
    )
    axis.grid(axis="y")
    axis.legend(loc="lower left", ncol=2, fontsize=8)
    panel_label(axis, "A")

    axis = axes[0, 1]
    width = 0.34
    x = np.arange(2)
    original_loc = original_det["localization_metrics"]
    purged_loc = purged_det["localization_metrics"]
    first = axis.bar(
        x - width / 2,
        [original_loc["ap50"], original_loc["ap50_95"]],
        width,
        color="white",
        edgecolor=ORANGE,
        hatch="///",
        linewidth=1.5,
        label="Original split (optimistic)",
    )
    second = axis.bar(
        x + width / 2,
        [purged_loc["ap50"], purged_loc["ap50_95"]],
        width,
        color=AMBER,
        alpha=0.72,
        edgecolor="#8D671A",
        linewidth=0.8,
        label="Purged sensitivity split",
    )
    axis.bar_label(first, fmt="%.3f", padding=3, fontsize=8)
    axis.bar_label(second, fmt="%.3f", padding=3, fontsize=8)
    axis.set(
        title="Detector localization",
        ylabel="Mean average precision",
        xticks=x,
        xticklabels=["AP@0.50", "AP@0.50:0.95"],
        ylim=(0, 0.82),
    )
    axis.grid(axis="y")
    axis.legend(loc="upper right", fontsize=8)
    panel_label(axis, "B")

    axis = axes[1, 0]
    original_shortcut = original_audit["feature_only_logistic_baseline"]["metrics"]["test"]["macro_f1"]
    purged_shortcut = purged_audit["feature_only_logistic_baseline"]["metrics"]["test"]["macro_f1"]
    original_hash = original_audit["cross_split_dhash"]
    purged_hash = purged_audit["cross_split_dhash"]
    indicators = [
        "Appearance-only\nmacro F1",
        "Median nearest\nblock SSIM",
        "Share with nearest\nSSIM ≥ 0.95",
    ]
    original_values = [
        original_shortcut,
        original_hash["blockwise_ssim_median"],
        original_hash["pairs_blockwise_ssim_at_or_above_0_95"] / original_hash["query_images"],
    ]
    purged_values = [
        purged_shortcut,
        purged_hash["blockwise_ssim_median"],
        purged_hash["pairs_blockwise_ssim_at_or_above_0_95"] / purged_hash["query_images"],
    ]
    x = np.arange(3)
    first = axis.bar(
        x - width / 2,
        original_values,
        width,
        color="white",
        edgecolor=NAVY,
        hatch="///",
        linewidth=1.5,
        label="Original split",
    )
    second = axis.bar(
        x + width / 2,
        purged_values,
        width,
        color=BLUE,
        alpha=0.72,
        edgecolor=NAVY,
        linewidth=0.8,
        label="Purged split",
    )
    axis.bar_label(first, fmt="%.3f", padding=3, fontsize=8)
    axis.bar_label(second, fmt="%.3f", padding=3, fontsize=8)
    axis.set(
        title="Shortcut and frame-proximity diagnostics",
        ylabel="Proportion or score",
        xticks=x,
        xticklabels=indicators,
        ylim=(0, 1.04),
    )
    axis.grid(axis="y")
    axis.legend(fontsize=8)
    panel_label(axis, "C")

    axis = axes[1, 1]
    axis.axis("off")
    rows = [
        ["Role", "Original export", "Purged sensitivity view"],
        ["Train / val / test", "1,032 / 110 / 93", "728 / 200 / 219"],
        ["Acquisition sequences", "3; all cross splits", "3; chronological within strata"],
        ["Boundary purge", "None", "2 source groups / side"],
        ["Independent prints", "Not verified", "Not verified"],
        ["Permitted claim", "Development result", "Temporal-overlap sensitivity"],
    ]
    table = axis.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.28, 0.32, 0.40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.1)
    table.scale(1.0, 1.58)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#C6CDD3")
        cell.set_linewidth(0.7)
        cell.set_facecolor("white" if row else "#EAF0F4")
        if row == 0:
            cell.set_text_props(weight="bold", color=INK)
    axis.set_title("Interpretation boundary", pad=8)
    panel_label(axis, "D")

    figure.suptitle(
        "Clinic-v2 sensitivity analysis: apparent performance versus temporal separation (seed 17)",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    save(figure, "vision_split_sensitivity_seed17")


def confusion(rows: list[dict[str, str]], names: list[str]) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=int)
    for row in rows:
        matrix[lookup[row["true_class"]], lookup[row["predicted_class"]]] += 1
    return matrix


def confusion_figure(sources: list[tuple[str, Path, str]], names: list[str]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.1, 8.1), constrained_layout=True)
    display_names = ["Background", "Spaghettification", "Underextrusion"]
    for panel, (axis, (title, path, color)) in enumerate(zip(axes.flat, sources)):
        matrix = confusion(load_rows(path), names)
        totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)
        cmap = LinearSegmentedColormap.from_list("white_accent", ["#FFFFFF", color])
        axis.imshow(normalized, cmap=cmap, vmin=0, vmax=1, alpha=0.34)
        for row in range(3):
            for column in range(3):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]}\n{normalized[row, column] * 100:.1f}%",
                    ha="center",
                    va="center",
                    color=INK,
                    fontsize=9.4,
                    fontweight="bold" if row == column else "normal",
                )
        axis.set(
            title=title,
            xlabel="Predicted class",
            ylabel="True class",
            xticks=np.arange(3),
            yticks=np.arange(3),
            xticklabels=display_names,
            yticklabels=display_names,
        )
        axis.tick_params(length=0)
        axis.tick_params(axis="x", labelrotation=18)
        axis.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        axis.grid(which="minor", color="#AEB6BF", linewidth=0.8)
        axis.grid(which="major", visible=False)
        panel_label(axis, chr(ord("A") + panel))
    figure.suptitle(
        "Image-level confusion matrices: original and purged Clinic-v2 views (seed 17)",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    save(figure, "vision_split_confusion_matrices_seed17")


def make_rows(payloads: list[tuple[str, str, str, dict]]) -> list[dict]:
    output = []
    for split, task, model, payload in payloads:
        shared, _ = image_metrics(payload, task)
        localization = payload.get("localization_metrics", {})
        output.append(
            {
                "split_view": split,
                "task": task,
                "model": model,
                "seed": 17,
                "test_images": payload["dataset"]["images"],
                "accuracy": shared["accuracy"],
                "balanced_accuracy": shared["balanced_accuracy"],
                "macro_f1": shared["macro_f1"],
                "ap50": localization.get("ap50", ""),
                "ap50_95": localization.get("ap50_95", ""),
                "median_latency_ms": payload["latency"]["median_ms"],
                "p95_latency_ms": payload["latency"]["p95_ms"],
                "checkpoint_mb": payload["model"]["size_mb"],
            }
        )
    return output


def write_outputs(rows: list[dict], source_paths: list[Path], detector_threshold: float) -> None:
    csv_path = METRICS_DIR / "vision_split_sensitivity_seed17.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": 1,
        "purpose": "post_hoc temporal-overlap sensitivity analysis; not independent-print validation",
        "interpretation": {
            "original": "Optimistic development split; frames from all three acquisition sequences cross partitions.",
            "purged": "Chronological within sequence/class strata with boundary purging; still only three sequences and class is partly sequence-confounded.",
        },
        "purged_detector_threshold_selected_on_validation": detector_threshold,
        "rows": rows,
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in source_paths
        ],
        "figure_files": sorted(
            str(path.resolve()) for path in FIGURES_DIR.glob("vision_split_*_seed17.*")
        ),
    }
    (METRICS_DIR / "vision_split_sensitivity_seed17.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    by_key = {(row["split_view"], row["task"]): row for row in rows}
    lines = [
        "# Clinic-v2 split sensitivity comparison (seed 17)",
        "",
        "The original export is an optimistic development split. The purged view is a temporal-overlap sensitivity analysis, not independent-print validation.",
        "",
        "| Split view | Task | n | Accuracy | Balanced accuracy | Macro F1 | AP50 | AP50:95 | Median latency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, task in [
        ("original", "classification"),
        ("original", "detection"),
        ("purged_chronological", "classification"),
        ("purged_chronological", "detection"),
    ]:
        row = by_key[(split, task)]
        ap50 = "—" if row["ap50"] == "" else f"{row['ap50']:.3f}"
        ap95 = "—" if row["ap50_95"] == "" else f"{row['ap50_95']:.3f}"
        lines.append(
            f"| {split.replace('_', ' ')} | {task} | {row['test_images']} | "
            f"{row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} | "
            f"{ap50} | {ap95} | {row['median_latency_ms']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            f"The purged detector confidence threshold ({detector_threshold:.2f}) was selected on validation data only and frozen before test evaluation.",
            "",
            "The two test sets differ in size and temporal construction. Their score difference is a sensitivity diagnostic, not a paired causal estimate.",
        ]
    )
    (TABLES_DIR / "vision_split_sensitivity_seed17.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    for directory in (METRICS_DIR, TABLES_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    configure()

    source_paths = [
        ORIGINAL_CLASSIFIER / "metrics.json",
        ORIGINAL_DETECTOR / "metrics.json",
        PURGED_CLASSIFIER / "metrics.json",
        PURGED_DETECTOR / "metrics.json",
        ORIGINAL_CLASSIFIER / "predictions.csv",
        ORIGINAL_DETECTOR / "predictions.csv",
        PURGED_CLASSIFIER / "predictions.csv",
        PURGED_DETECTOR / "predictions.csv",
        METRICS_DIR / "clinic_v2_shortcut_audit.json",
        METRICS_DIR / "clinic_v2_purged_shortcut_audit.json",
        ROOT / "data/manifests/clinic_v2_purged_summary.json",
        PURGED_DETECTOR.parent / "validation_operating_point/operating_point.json",
    ]
    original_cls, original_det, purged_cls, purged_det = [load_json(path) for path in source_paths[:4]]
    original_audit = load_json(source_paths[8])
    purged_audit = load_json(source_paths[9])
    operating_point = load_json(source_paths[11])
    threshold = float(operating_point["selected_threshold"])

    payloads = [
        ("original", "classification", "YOLO11n-cls", original_cls),
        ("original", "detection", "YOLO11n", original_det),
        ("purged_chronological", "classification", "YOLO11n-cls", purged_cls),
        ("purged_chronological", "detection", "YOLO11n", purged_det),
    ]
    rows = make_rows(payloads)
    metric_figure(original_cls, original_det, purged_cls, purged_det, original_audit, purged_audit)
    confusion_figure(
        [
            ("Original classification (optimistic)", source_paths[4], NAVY),
            ("Original detection (optimistic)", source_paths[5], ORANGE),
            ("Purged classification", source_paths[6], BLUE),
            ("Purged detection", source_paths[7], AMBER),
        ],
        original_cls["dataset"]["classes"],
    )
    write_outputs(rows, source_paths, threshold)
    print(
        json.dumps(
            {
                "metrics": str((METRICS_DIR / "vision_split_sensitivity_seed17.json").resolve()),
                "table": str((TABLES_DIR / "vision_split_sensitivity_seed17.md").resolve()),
                "figures": sorted(path.name for path in FIGURES_DIR.glob("vision_split_*_seed17.*")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
