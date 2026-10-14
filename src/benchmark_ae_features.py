"""Reproducible development benchmark for the archived AE feature table.

The source rows do not have verified print/job identifiers. This script therefore
uses one frozen stratified row split and reports development sensitivity only; it
does not present row resampling as independent-print validation.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SOURCE = ROOT / "data/recorded/ae_overflow.csv"
RUNS = ROOT / "runs/acoustic/ae_overflow_feature_benchmark"
METRICS = ROOT / "reports/metrics"
TABLES = ROOT / "reports/tables"
FIGURES = ROOT / "reports/figures"
FEATURES = ["amplitude", "duration", "energy", "rms", "rise_time", "counts"]
SEEDS = (17, 29, 43)
SPLIT_SEED = 17
COLORS = {
    "Logistic regression": "#153A59",
    "Random forest": "#147D74",
    "XGBoost": "#C85B21",
    "CatBoost": "#7B4F9D",
    "Small MLP": "#C69214",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ece(y_true: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        include = (probability >= low) & (probability < high if high < 1.0 else probability <= high)
        if not include.any():
            continue
        value += include.mean() * abs(float(y_true[include].mean()) - float(probability[include].mean()))
    return float(value)


def operating_point(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, dict]:
    best = None
    for threshold in np.linspace(0.05, 0.95, 181):
        prediction = (probability >= threshold).astype(int)
        score = {
            "threshold": float(threshold),
            "macro_f1": float(f1_score(y_true, prediction, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        }
        key = (score["macro_f1"], score["balanced_accuracy"], -score["threshold"])
        if best is None or key > best[0]:
            best = (key, score)
    assert best is not None
    return best[1]["threshold"], best[1]


def model_factory(seed: int) -> dict[str, object]:
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.035,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=seed,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=350,
            depth=5,
            learning_rate=0.035,
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
            thread_count=-1,
        ),
        "Small MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=1e-3,
                batch_size=32,
                learning_rate_init=1e-3,
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=35,
                random_state=seed,
            ),
        ),
    }


def evaluate(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    result = {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro")),
        "defect_precision": float(precision_score(y_true, prediction, zero_division=0)),
        "defect_sensitivity": float(recall_score(y_true, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "defect_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece_15bin": ece(y_true, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return result, prediction


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    keys = [
        "accuracy", "balanced_accuracy", "macro_f1", "defect_precision",
        "defect_sensitivity", "specificity", "defect_f1", "auroc", "brier",
        "ece_15bin", "latency_ms", "threshold",
    ]
    for model in COLORS:
        subset = [row for row in rows if row["model"] == model]
        summary = {"model": model, "seeds": len(subset)}
        for key in keys:
            values = np.asarray([row[key] for row in subset], dtype=float)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_sd"] = float(values.std(ddof=1))
        output.append(summary)
    return output


def make_figure(rows: list[dict], summaries: list[dict], y_test: np.ndarray) -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9.2, "axes.titlesize": 11,
        "axes.titleweight": "bold", "grid.color": "#D9DEE3", "grid.alpha": 0.85,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.1, 8.0), constrained_layout=True)
    models = list(COLORS)
    x = np.arange(len(models))

    ax = axes[0, 0]
    width = 0.25
    for offset, (key, label) in zip((-width, 0, width), (("macro_f1", "Macro F1"), ("defect_sensitivity", "Sensitivity"), ("specificity", "Specificity"))):
        means = [next(s for s in summaries if s["model"] == model)[f"{key}_mean"] for model in models]
        sds = [next(s for s in summaries if s["model"] == model)[f"{key}_sd"] for model in models]
        ax.bar(x + offset, means, width, yerr=sds, capsize=2.5, label=label, alpha=0.76)
    ax.set(title="Development discrimination by model", ylabel="Mean score ± SD", xticks=x, xticklabels=models, ylim=(0.45, 1.02))
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(axis="y")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    probability_label_offsets = {
        "Logistic regression": (6, 4),
        "Random forest": (7, 9),
        "CatBoost": (-34, -13),
        "Small MLP": (8, 8),
        "XGBoost": (8, -5),
    }
    for model in models:
        summary = next(s for s in summaries if s["model"] == model)
        ax.scatter(summary["brier_mean"], summary["ece_15bin_mean"], s=75, color=COLORS[model], label=model)
        ax.annotate(model, (summary["brier_mean"], summary["ece_15bin_mean"]), xytext=probability_label_offsets[model], textcoords="offset points", fontsize=8, color=COLORS[model])
    ax.set(title="Probability diagnostics", xlabel="Brier score (lower is better)", ylabel="ECE, 15 bins (lower is better)")
    ax.grid()

    ax = axes[1, 0]
    best_model = max(summaries, key=lambda item: item["macro_f1_mean"])["model"]
    best_rows = [row for row in rows if row["model"] == best_model]
    for row in best_rows:
        prob_true, prob_pred = calibration_curve(y_test, np.asarray(row["test_probabilities"]), n_bins=8, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", lw=1.4, alpha=0.75, label=f"seed {row['seed']}")
    ax.plot([0, 1], [0, 1], ls="--", color="#7C858D", lw=1)
    ax.set(title=f"Reliability of selected development model: {best_model}", xlabel="Mean predicted probability", ylabel="Observed defect fraction", xlim=(0, 1), ylim=(0, 1))
    ax.grid()
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    macro = [next(s for s in summaries if s["model"] == model)["macro_f1_mean"] for model in models]
    latency = [next(s for s in summaries if s["model"] == model)["latency_ms_mean"] for model in models]
    latency_label_offsets = {
        "Logistic regression": (8, 1),
        "Random forest": (8, 5),
        "XGBoost": (8, -13),
        "CatBoost": (8, 10),
        "Small MLP": (8, 20),
    }
    for model, xval, yval in zip(models, latency, macro):
        ax.scatter(xval, yval, s=78, color=COLORS[model])
        ax.annotate(model, (xval, yval), xytext=latency_label_offsets[model], textcoords="offset points", fontsize=8, color=COLORS[model])
    ax.set(title="Performance–latency trade-off", xlabel="Median batch-one inference (ms)", ylabel="Mean macro F1", ylim=(0.45, 1.02))
    ax.grid()

    fig.suptitle("Archived acoustic-emission feature benchmark (row-level development split)", fontsize=14, fontweight="bold")
    for suffix, kwargs in (("png", {"dpi": 600}), ("pdf", {})):
        fig.savefig(FIGURES / f"ae_feature_benchmark.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def main() -> None:
    global SOURCE, RUNS, METRICS, TABLES, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts/ae_rerun")
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    output = args.output_root.resolve()
    RUNS = output / "runs/acoustic/ae_overflow_feature_benchmark"
    METRICS, TABLES, FIGURES = (output / "reports" / name for name in ("metrics", "tables", "figures"))
    for directory in (RUNS, METRICS, TABLES, FIGURES):
        directory.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SOURCE)
    data = data.loc[:, ~data.columns.astype(str).str.startswith("Unnamed")]
    data = data[FEATURES + ["label"]].dropna().copy()
    data["label"] = data["label"].astype(int)
    feature_hashes = data[FEATURES].astype(str).agg("|".join, axis=1).map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    conflict_count = int(pd.DataFrame({"hash": feature_hashes, "label": data["label"]}).groupby("hash")["label"].nunique().gt(1).sum())

    all_indices = np.arange(len(data))
    train_idx, holdout_idx = train_test_split(all_indices, test_size=0.40, stratify=data["label"], random_state=SPLIT_SEED)
    val_idx, test_idx = train_test_split(holdout_idx, test_size=0.50, stratify=data.iloc[holdout_idx]["label"], random_state=SPLIT_SEED)
    split_rows = []
    for split, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        for index in indices:
            split_rows.append({"source_row": int(index), "split": split, "label": int(data.iloc[index]["label"]), "feature_hash": feature_hashes.iloc[index]})
    with (RUNS / "frozen_row_split.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(split_rows[0]))
        writer.writeheader()
        writer.writerows(split_rows)

    x = data[FEATURES].to_numpy(dtype=np.float64)
    y = data["label"].to_numpy(dtype=int)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    rows = []
    prediction_rows = []
    for seed in SEEDS:
        for model_name, model in model_factory(seed).items():
            fitted = clone(model)
            started = time.perf_counter()
            fitted.fit(x_train, y_train)
            train_seconds = time.perf_counter() - started
            val_probability = fitted.predict_proba(x_val)[:, 1]
            threshold, val_selection = operating_point(y_val, val_probability)
            test_probability = fitted.predict_proba(x_test)[:, 1]

            for _ in range(20):
                fitted.predict_proba(x_test[:1])
            latencies = []
            for sample in x_test[:100]:
                tick = time.perf_counter()
                fitted.predict_proba(sample.reshape(1, -1))
                latencies.append((time.perf_counter() - tick) * 1000.0)
            result, prediction = evaluate(y_test, test_probability, threshold)
            row = {
                "model": model_name, "seed": seed, "threshold": threshold,
                "validation_macro_f1": val_selection["macro_f1"],
                "validation_balanced_accuracy": val_selection["balanced_accuracy"],
                "training_seconds": train_seconds,
                "latency_ms": float(np.median(latencies)),
                **result,
                "test_probabilities": test_probability.tolist(),
            }
            rows.append(row)
            model_dir = RUNS / f"{model_name.lower().replace(' ', '_')}_seed{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(fitted, model_dir / "model.joblib")
            (model_dir / "metrics.json").write_text(json.dumps({k: v for k, v in row.items() if k != "test_probabilities"}, indent=2), encoding="utf-8")
            for index, probability, predicted in zip(test_idx, test_probability, prediction):
                prediction_rows.append({"model": model_name, "seed": seed, "source_row": int(index), "y_true": int(y[index]), "probability": float(probability), "prediction": int(predicted), "threshold": threshold})

    summaries = summarize(rows)
    make_figure(rows, summaries, y_test)

    serial_rows = [{k: v for k, v in row.items() if k != "test_probabilities"} for row in rows]
    with (METRICS / "ae_feature_benchmark_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(serial_rows[0]))
        writer.writeheader(); writer.writerows(serial_rows)
    with (RUNS / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader(); writer.writerows(prediction_rows)

    payload = {
        "schema_version": 1,
        "scientific_use": "Row-level development sensitivity only; no print/job identifiers are available.",
        "source": {"path": str(SOURCE.resolve()), "sha256": sha256(SOURCE), "rows": len(data), "features": FEATURES, "class_counts": data["label"].value_counts().sort_index().to_dict(), "cross_label_feature_conflicts": conflict_count},
        "split": {"seed": SPLIT_SEED, "train": len(train_idx), "validation": len(val_idx), "test": len(test_idx), "manifest": str((RUNS / "frozen_row_split.csv").resolve())},
        "model_seeds": list(SEEDS),
        "rows": serial_rows,
        "summaries": summaries,
        "guard": "Do not call these rows independent prints or use row-level resampling to imply manufacturing generalization.",
    }
    (METRICS / "ae_feature_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Acoustic-emission engineered-feature development benchmark", "",
        "All values are mean ± sample SD across model seeds 17, 29, and 43 on one frozen 168-row test split. The 840-row source has no verified print/job identifiers, so this is development sensitivity only.", "",
        "| Model | Balanced accuracy | Macro F1 | Defect sensitivity | Specificity | AUROC | Brier | ECE | Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in sorted(summaries, key=lambda item: item["macro_f1_mean"], reverse=True):
        lines.append(
            f"| {summary['model']} | {summary['balanced_accuracy_mean']:.3f} ± {summary['balanced_accuracy_sd']:.3f} | "
            f"{summary['macro_f1_mean']:.3f} ± {summary['macro_f1_sd']:.3f} | {summary['defect_sensitivity_mean']:.3f} ± {summary['defect_sensitivity_sd']:.3f} | "
            f"{summary['specificity_mean']:.3f} ± {summary['specificity_sd']:.3f} | {summary['auroc_mean']:.3f} ± {summary['auroc_sd']:.3f} | "
            f"{summary['brier_mean']:.3f} ± {summary['brier_sd']:.3f} | {summary['ece_15bin_mean']:.3f} ± {summary['ece_15bin_sd']:.3f} | "
            f"{summary['latency_ms_mean']:.3f} ± {summary['latency_ms_sd']:.3f} ms |"
        )
    lines += ["", "Thresholds were selected on the validation split separately for each fitted seed. Test data were not used for model or threshold selection.", "", "Independent grouped acquisition is still required before reporting print-level confidence intervals or manufacturing generalization."]
    (TABLES / "ae_feature_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(data), "test_rows": len(test_idx), "models": len(summaries), "report": str((METRICS / "ae_feature_benchmark.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
