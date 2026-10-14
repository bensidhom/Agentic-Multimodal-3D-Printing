"""Synthetic paired replay using archived vision and AE held-out score distributions.

The two source datasets were not recorded on the same print jobs. Pairing is
therefore simulated by class-conditional resampling and must never be presented
as empirical multimodal validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/fusion/synthetic_paired_replay"
METRICS = ROOT / "reports/metrics"
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures"
SEEDS = (17, 29, 43)
N_TRAIN, N_VAL, N_TEST = 6000, 2000, 2000
DEFECT_PREVALENCE = 0.30
POLICIES = (
    "Vision only", "AE only", "OR", "AND",
    "Logistic late fusion", "Gradient-boosted late fusion",
)
COLORS = {
    "Vision only": "#153A59",
    "AE only": "#147D74",
    "OR": "#C85B21",
    "AND": "#7B4F9D",
    "Logistic late fusion": "#C69214",
    "Gradient-boosted late fusion": "#B23A48",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        prediction = (score >= threshold).astype(int)
        macro = f1_score(y_true, prediction, average="macro")
        balanced = balanced_accuracy_score(y_true, prediction)
        key = (macro, balanced, -threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return best[1]


def metric_row(y_true: np.ndarray, prediction: np.ndarray, score: np.ndarray | None = None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    row = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "defect_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "defect_sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "false_positive_rate": float(fp / (tn + fp)),
        "miss_rate": float(fn / (fn + tp)),
        "defect_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if score is not None:
        row["auroc"] = float(roc_auc_score(y_true, score))
        row["brier"] = float(brier_score_loss(y_true, score))
    else:
        row["auroc"] = ""
        row["brier"] = ""
    return row


def source_frames(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    vision_path = ROOT / f"runs/detection/clinic_v2_purged_yolo11n_detect_seed{seed}/test_evaluation/predictions.csv"
    ae_path = ROOT / "runs/acoustic/ae_overflow_feature_benchmark/test_predictions.csv"
    vision = pd.read_csv(vision_path)
    vision = pd.DataFrame({
        "y": (vision["true_class"] != "background").astype(int),
        "score": 1.0 - vision["score_background"].astype(float),
        "decision": (vision["predicted_class"] != "background").astype(int),
    })
    ae_all = pd.read_csv(ae_path)
    ae = ae_all[(ae_all["model"] == "Small MLP") & (ae_all["seed"] == seed)].copy()
    ae = pd.DataFrame({
        "y": ae["y_true"].astype(int),
        "score": ae["probability"].astype(float),
        "decision": ae["prediction"].astype(int),
    })
    return vision, ae, [vision_path, ae_path]


def simulate_split(
    rng: np.random.Generator,
    vision: pd.DataFrame,
    ae: pd.DataFrame,
    n: int,
) -> pd.DataFrame:
    y = rng.binomial(1, DEFECT_PREVALENCE, size=n)
    output = {"y": y}
    for prefix, frame in (("vision", vision), ("ae", ae)):
        scores = np.empty(n, dtype=float)
        decisions = np.empty(n, dtype=int)
        for label in (0, 1):
            positions = np.flatnonzero(y == label)
            source = frame[frame["y"] == label]
            chosen = rng.integers(0, len(source), size=len(positions))
            scores[positions] = source["score"].to_numpy()[chosen]
            decisions[positions] = source["decision"].to_numpy()[chosen]
        output[f"{prefix}_score"] = scores
        output[f"{prefix}_decision"] = decisions
    return pd.DataFrame(output)


def run_seed(seed: int) -> tuple[list[dict], list[Path]]:
    vision, ae, sources = source_frames(seed)
    rng = np.random.default_rng(seed)
    train = simulate_split(rng, vision, ae, N_TRAIN)
    val = simulate_split(rng, vision, ae, N_VAL)
    test = simulate_split(rng, vision, ae, N_TEST)
    features = ["vision_score", "ae_score"]

    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    boosted = GradientBoostingClassifier(n_estimators=160, learning_rate=0.035, max_depth=2, random_state=seed)
    logistic.fit(train[features], train["y"])
    boosted.fit(train[features], train["y"])
    val_logistic = logistic.predict_proba(val[features])[:, 1]
    val_boosted = boosted.predict_proba(val[features])[:, 1]
    logistic_threshold = select_threshold(val["y"].to_numpy(), val_logistic)
    boosted_threshold = select_threshold(val["y"].to_numpy(), val_boosted)
    test_logistic = logistic.predict_proba(test[features])[:, 1]
    test_boosted = boosted.predict_proba(test[features])[:, 1]

    policies: dict[str, tuple[np.ndarray, np.ndarray | None, float | str]] = {
        "Vision only": (test["vision_decision"].to_numpy(), test["vision_score"].to_numpy(), "source operating point"),
        "AE only": (test["ae_decision"].to_numpy(), test["ae_score"].to_numpy(), "source operating point"),
        "OR": ((test["vision_decision"] | test["ae_decision"]).to_numpy(), None, "logical OR"),
        "AND": ((test["vision_decision"] & test["ae_decision"]).to_numpy(), None, "logical AND"),
        "Logistic late fusion": ((test_logistic >= logistic_threshold).astype(int), test_logistic, logistic_threshold),
        "Gradient-boosted late fusion": ((test_boosted >= boosted_threshold).astype(int), test_boosted, boosted_threshold),
    }
    rows = []
    for policy in POLICIES:
        prediction, score, threshold = policies[policy]
        rows.append({"policy": policy, "seed": seed, "threshold": threshold, **metric_row(test["y"].to_numpy(), prediction, score)})

    seed_dir = RUN / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    test.to_csv(seed_dir / "synthetic_test_pairs.csv", index=False)
    (seed_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows, sources


def summarize(rows: list[dict]) -> list[dict]:
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "defect_precision", "defect_sensitivity", "specificity", "false_positive_rate", "miss_rate", "defect_f1"]
    output = []
    for policy in POLICIES:
        subset = [row for row in rows if row["policy"] == policy]
        summary = {"policy": policy, "seeds": len(subset)}
        for key in keys:
            values = np.asarray([row[key] for row in subset], dtype=float)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_sd"] = float(values.std(ddof=1))
        for key in ("auroc", "brier"):
            values = [float(row[key]) for row in subset if row[key] != ""]
            summary[f"{key}_mean"] = float(np.mean(values)) if values else ""
            summary[f"{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else ""
        output.append(summary)
    return output


def make_figure(rows: list[dict], summaries: list[dict]) -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9.2, "axes.titlesize": 11,
        "axes.titleweight": "bold", "grid.color": "#D9DEE3", "grid.alpha": 0.85,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 8.1), constrained_layout=True)
    x = np.arange(len(POLICIES))
    labels = ["Vision only", "AE only", "OR", "AND", "Logistic\nfusion", "Boosted\nfusion"]

    ax = axes[0, 0]
    width = 0.36
    for offset, key, label in ((-width/2, "defect_sensitivity", "Sensitivity"), (width/2, "specificity", "Specificity")):
        means = [next(item for item in summaries if item["policy"] == policy)[f"{key}_mean"] for policy in POLICIES]
        sds = [next(item for item in summaries if item["policy"] == policy)[f"{key}_sd"] for policy in POLICIES]
        ax.bar(x + offset, means, width, yerr=sds, capsize=2.5, alpha=0.78, label=label)
    ax.set(title="Detection and rejection behavior", ylabel="Mean score ± SD", xticks=x, xticklabels=labels, ylim=(0, 1.03))
    ax.tick_params(axis="x", labelrotation=12)
    ax.grid(axis="y"); ax.legend(fontsize=8)

    ax = axes[0, 1]
    for index, policy in enumerate(POLICIES):
        subset = [row for row in rows if row["policy"] == policy]
        for row, jitter in zip(subset, (-0.08, 0.0, 0.08)):
            ax.scatter(index + jitter, row["macro_f1"], s=42, color=COLORS[policy], edgecolor="white", linewidth=0.7)
            ax.text(index + jitter, row["macro_f1"] + 0.009, str(row["seed"]), ha="center", fontsize=7, color=COLORS[policy])
        summary = next(item for item in summaries if item["policy"] == policy)
        ax.errorbar(index, summary["macro_f1_mean"], yerr=summary["macro_f1_sd"], fmt="D", color="#20252A", mfc="white", capsize=4, ms=5.5)
    ax.set(title="Policy macro F1 by source-model seed", ylabel="Synthetic test macro F1", xticks=x, xticklabels=labels, ylim=(0.45, 1.02))
    ax.tick_params(axis="x", labelrotation=12); ax.grid(axis="y")

    ax = axes[1, 0]
    width = 0.36
    for offset, key, label in ((-width/2, "false_positive_rate", "False-positive rate"), (width/2, "miss_rate", "Miss rate")):
        means = [next(item for item in summaries if item["policy"] == policy)[f"{key}_mean"] for policy in POLICIES]
        sds = [next(item for item in summaries if item["policy"] == policy)[f"{key}_sd"] for policy in POLICIES]
        ax.bar(x + offset, means, width, yerr=sds, capsize=2.5, alpha=0.78, label=label)
    ax.set(title="Safety-relevant error trade-off", ylabel="Mean rate ± SD", xticks=x, xticklabels=labels, ylim=(0, 0.65))
    ax.tick_params(axis="x", labelrotation=12); ax.grid(axis="y"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    for policy in POLICIES:
        summary = next(item for item in summaries if item["policy"] == policy)
        ax.scatter(summary["false_positive_rate_mean"], summary["defect_sensitivity_mean"], s=78, color=COLORS[policy])
        offset = (6, 4) if policy not in ("OR", "AND") else ((6, -13) if policy == "OR" else (6, 8))
        ax.annotate(policy, (summary["false_positive_rate_mean"], summary["defect_sensitivity_mean"]), xytext=offset, textcoords="offset points", fontsize=8, color=COLORS[policy])
    ax.set(title="Operating trade-off", xlabel="False-positive rate", ylabel="Defect sensitivity", xlim=(0, 0.55), ylim=(0.4, 1.03))
    ax.grid()

    fig.suptitle("Synthetic multimodal paired replay from archived held-out score distributions", fontsize=14, fontweight="bold")
    fig.savefig(FIGURES / "multimodal_synthetic_fusion.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "multimodal_synthetic_fusion.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    for directory in (RUN, METRICS, TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    source_paths: set[Path] = set()
    for seed in SEEDS:
        seed_rows, paths = run_seed(seed)
        rows.extend(seed_rows)
        source_paths.update(paths)
    summaries = summarize(rows)
    make_figure(rows, summaries)

    with (METRICS / "multimodal_synthetic_fusion_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "purpose": "synthetic paired multimodal policy comparison",
        "scientific_use": "Simulation only; not evidence from physically paired multimodal print jobs.",
        "pairing_assumption": "Vision and AE scores are sampled independently conditional on the binary ground-truth event label.",
        "scenario": {"defect_prevalence": DEFECT_PREVALENCE, "train_events": N_TRAIN, "validation_events": N_VAL, "test_events": N_TEST, "seeds": list(SEEDS)},
        "source_files": [{"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(source_paths)],
        "rows": rows,
        "summaries": summaries,
        "guard": "Do not replace a paired print-job ablation with this result. It tests policy behavior under an explicit score-pairing model.",
    }
    (METRICS / "multimodal_synthetic_fusion.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Synthetic multimodal paired replay", "",
        f"Values are mean ± sample SD across source-model seeds on {N_TEST:,} simulated test events per seed at {DEFECT_PREVALENCE:.0%} defect prevalence. Vision and AE held-out scores were paired by independent class-conditional resampling. This is simulation, not paired physical validation.", "",
        "| Policy | Balanced accuracy | Macro F1 | Sensitivity | Specificity | False-positive rate | Miss rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in sorted(summaries, key=lambda item: item["macro_f1_mean"], reverse=True):
        lines.append(
            f"| {summary['policy']} | {summary['balanced_accuracy_mean']:.3f} ± {summary['balanced_accuracy_sd']:.3f} | {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_sd']:.3f} | "
            f"{summary['defect_sensitivity_mean']:.3f} ± {summary['defect_sensitivity_sd']:.3f} | {summary['specificity_mean']:.3f} ± {summary['specificity_sd']:.3f} | "
            f"{summary['false_positive_rate_mean']:.3f} ± {summary['false_positive_rate_sd']:.3f} | {summary['miss_rate_mean']:.3f} ± {summary['miss_rate_sd']:.3f} |"
        )
    lines += ["", "Late-fusion thresholds were selected on the synthetic validation split only. A paired print-job experiment remains required."]
    (TABLES / "multimodal_synthetic_fusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "policies": len(summaries), "report": str((METRICS / "multimodal_synthetic_fusion.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
