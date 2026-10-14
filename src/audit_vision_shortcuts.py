"""Post-hoc shortcut and near-duplicate audit for the Clinic v2 image corpus."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import ExifTags, Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"
FEATURE_NAMES = [
    "log_file_bytes",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "gray_q10",
    "gray_q25",
    "gray_q50",
    "gray_q75",
    "gray_q90",
    "horizontal_edge_mean",
    "vertical_edge_mean",
    *[f"gray_hist_{index:02d}" for index in range(16)],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "clinic_v2_task_comparison_manifest.csv",
    )
    parser.add_argument("--output-prefix", default="clinic_v2_shortcut")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(gray: np.ndarray) -> int:
    image = Image.fromarray(gray.astype(np.uint8)).resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(image)
    bits = values[:, 1:] > values[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def image_features(path: Path) -> tuple[np.ndarray, int, str | None]:
    with Image.open(path) as image:
        exif = image.getexif()
        make = exif.get(next((key for key, value in ExifTags.TAGS.items() if value == "Make"), -1))
        model = exif.get(next((key for key, value in ExifTags.TAGS.items() if value == "Model"), -1))
        camera = " / ".join(str(value).strip() for value in (make, model) if value) or None
        rgb_image = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    gray = np.dot(rgb, np.asarray([0.299, 0.587, 0.114], dtype=np.float32))
    channel_mean = rgb.mean(axis=(0, 1))
    channel_std = rgb.std(axis=(0, 1))
    quantiles = np.quantile(gray, [0.10, 0.25, 0.50, 0.75, 0.90])
    horizontal = np.abs(np.diff(gray, axis=1)).mean()
    vertical = np.abs(np.diff(gray, axis=0)).mean()
    histogram, _ = np.histogram(gray, bins=16, range=(0.0, 1.0), density=False)
    histogram = histogram.astype(float) / histogram.sum()
    features = np.concatenate(
        (
            [np.log1p(path.stat().st_size)],
            channel_mean,
            channel_std,
            quantiles,
            [horizontal, vertical],
            histogram,
        )
    )
    gray_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return features.astype(float), dhash(gray_uint8), camera


def blockwise_ssim(left_gray: np.ndarray, right_gray: np.ndarray, blocks: int = 8) -> float:
    """Compute mean scalar SSIM over a regular block grid without extra dependencies."""
    c1 = 0.01**2
    c2 = 0.03**2
    height, width = left_gray.shape
    values: list[float] = []
    for row in range(blocks):
        for column in range(blocks):
            y0, y1 = row * height // blocks, (row + 1) * height // blocks
            x0, x1 = column * width // blocks, (column + 1) * width // blocks
            left = left_gray[y0:y1, x0:x1]
            right = right_gray[y0:y1, x0:x1]
            left_mean = float(left.mean())
            right_mean = float(right.mean())
            left_var = float(left.var())
            right_var = float(right.var())
            covariance = float(((left - left_mean) * (right - right_mean)).mean())
            numerator = (2 * left_mean * right_mean + c1) * (2 * covariance + c2)
            denominator = (left_mean**2 + right_mean**2 + c1) * (left_var + right_var + c2)
            values.append(numerator / denominator if denominator else 1.0)
    return float(np.mean(values))


def image_similarity(left_path: Path, right_path: Path) -> dict[str, float]:
    arrays = []
    for path in (left_path, right_path):
        with Image.open(path) as image:
            resized = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        arrays.append(np.asarray(resized, dtype=np.float32) / 255.0)
    left, right = arrays
    difference = np.abs(left - right)
    left_gray = np.dot(left, np.asarray([0.299, 0.587, 0.114], dtype=np.float32))
    right_gray = np.dot(right, np.asarray([0.299, 0.587, 0.114], dtype=np.float32))
    return {
        "rgb_mae": float(difference.mean()),
        "rgb_rmse": float(np.sqrt(np.mean((left - right) ** 2))),
        "changed_pixel_fraction_gt_0_10": float((difference.max(axis=2) > 0.10).mean()),
        "blockwise_ssim": blockwise_ssim(left_gray, right_gray),
    }


def nearest_hashes(reference: list[dict], query: list[dict]) -> list[dict]:
    output: list[dict] = []
    for row in query:
        distances = np.asarray([(row["dhash"] ^ item["dhash"]).bit_count() for item in reference])
        index = int(distances.argmin())
        match = reference[index]
        comparison = {
            "split": row["split"],
            "query_path": row["path"],
            "query_class": row["target"],
            "nearest_train_path": match["path"],
            "nearest_train_class": match["target"],
            "hamming_distance": int(distances[index]),
            "same_class": row["target"] == match["target"],
        }
        comparison.update(image_similarity(Path(row["path"]), Path(match["path"])))
        output.append(comparison)
    return output


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        manifest_rows = list(csv.DictReader(stream))
    records: list[dict] = []
    cameras: Counter[str] = Counter()
    for index, row in enumerate(manifest_rows, start=1):
        path = Path(row["image_path"])
        features, hash_value, camera = image_features(path)
        if camera:
            cameras[camera] += 1
        records.append(
            {
                "split": row["split"],
                "path": str(path),
                "sha256": row["image_sha256"],
                "source_group_id": row["source_group_id"],
                "target": row["classification_target"],
                "features": features,
                "dhash": hash_value,
                "camera": camera or "",
            }
        )
        if index % 250 == 0:
            print(f"processed {index}/{len(manifest_rows)} images")

    classes = sorted({row["target"] for row in records})
    class_to_index = {name: index for index, name in enumerate(classes)}
    by_split = {split: [row for row in records if row["split"] == split] for split in ("train", "val", "test")}
    train_x = np.vstack([row["features"] for row in by_split["train"]])
    train_y = np.asarray([class_to_index[row["target"]] for row in by_split["train"]])
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=17,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    baseline: dict[str, dict[str, float]] = {}
    for split in ("val", "test"):
        x = np.vstack([row["features"] for row in by_split[split]])
        y = np.asarray([class_to_index[row["target"]] for row in by_split[split]])
        predicted = model.predict(x)
        baseline[split] = {
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        }

    nearest = nearest_hashes(by_split["train"], by_split["val"] + by_split["test"])
    nearest_path = METRICS_DIR / f"{args.output_prefix}_nearest_hashes.csv"
    with nearest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(nearest[0]))
        writer.writeheader()
        writer.writerows(nearest)

    feature_path = METRICS_DIR / f"{args.output_prefix}_features.csv"
    with feature_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["split", "path", "sha256", "source_group_id", "target", "dhash_hex", "camera", *FEATURE_NAMES]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            values = {
                "split": row["split"],
                "path": row["path"],
                "sha256": row["sha256"],
                "source_group_id": row["source_group_id"],
                "target": row["target"],
                "dhash_hex": f"{row['dhash']:016x}",
                "camera": row["camera"],
            }
            values.update({name: float(value) for name, value in zip(FEATURE_NAMES, row["features"], strict=True)})
            writer.writerow(values)

    distances = np.asarray([row["hamming_distance"] for row in nearest])
    same_class = np.asarray([row["same_class"] for row in nearest], dtype=bool)
    rgb_mae = np.asarray([row["rgb_mae"] for row in nearest], dtype=float)
    structural = np.asarray([row["blockwise_ssim"] for row in nearest], dtype=float)
    changed = np.asarray([row["changed_pixel_fraction_gt_0_10"] for row in nearest], dtype=float)
    report = {
        "schema_version": 1,
        "purpose": "post_hoc_shortcut_diagnostic_after_seed17_perfect_test_result",
        "interpretation_guard": "Global appearance features can encode genuine defect morphology and acquisition artifacts; high performance is a warning signal, not proof of leakage. Perceptual-hash proximity is not identity.",
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "classes": classes,
        "feature_only_logistic_baseline": {
            "features": FEATURE_NAMES,
            "fixed_model": "StandardScaler + class-balanced multinomial logistic regression, C=1.0",
            "metrics": baseline,
        },
        "cross_split_dhash": {
            "query_images": int(distances.size),
            "minimum_distance": int(distances.min()),
            "median_distance": float(np.median(distances)),
            "p10_distance": float(np.quantile(distances, 0.10)),
            "pairs_at_or_below_4_bits": int((distances <= 4).sum()),
            "pairs_at_or_below_8_bits": int((distances <= 8).sum()),
            "nearest_neighbor_same_class_fraction": float(same_class.mean()),
            "rgb_mae_median": float(np.median(rgb_mae)),
            "rgb_mae_p10": float(np.quantile(rgb_mae, 0.10)),
            "blockwise_ssim_median": float(np.median(structural)),
            "pairs_blockwise_ssim_at_or_above_0_95": int((structural >= 0.95).sum()),
            "pairs_blockwise_ssim_at_or_above_0_98": int((structural >= 0.98).sum()),
            "pairs_rgb_mae_at_or_below_0_02": int((rgb_mae <= 0.02).sum()),
            "pairs_changed_pixel_fraction_at_or_below_0_05": int((changed <= 0.05).sum()),
            "rows_csv": str(nearest_path.resolve()),
        },
        "exif_camera_counts": dict(cameras),
        "feature_rows_csv": str(feature_path.resolve()),
        "external_domain_gate": {
            "clinic_to_hu_accuracy": "prohibited",
            "reason": "Non-overlapping label spaces make direct transfer accuracy semantically invalid.",
            "hu_classes": ["Cracking", "Layer_shifting", "Off_platform", "Stringing", "Warping"],
            "clinic_classes": classes,
        },
    }
    report_path = METRICS_DIR / f"{args.output_prefix}_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "grid.color": "#D8DDE3",
            "grid.alpha": 0.8,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    labels = ["Accuracy", "Balanced accuracy", "Macro F1"]
    x = np.arange(len(metrics))
    width = 0.34
    val_values = [baseline["val"][name] for name in metrics]
    test_values = [baseline["test"][name] for name in metrics]
    val_bars = axes[0].bar(x - width / 2, val_values, width, color="#0B3C5D", label="Validation")
    test_bars = axes[0].bar(x + width / 2, test_values, width, color="#D35400", label="Test")
    axes[0].bar_label(val_bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].bar_label(test_bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].set(title="Global-appearance-only baseline", ylabel="Score", xticks=x, xticklabels=labels, ylim=(0, 1.08))
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    axes[0].text(-0.12, 1.08, "A", transform=axes[0].transAxes, fontsize=12, fontweight="bold")

    bins = np.arange(distances.min() - 0.5, distances.max() + 1.5, 1)
    axes[1].hist(distances, bins=bins, color="#147D74", edgecolor="white", linewidth=0.5)
    axes[1].axvline(4, color="#D35400", ls="--", lw=1.8, label="Near-duplicate warning (≤4 bits)")
    axes[1].set(title="Nearest train-image perceptual distance", xlabel="64-bit dHash Hamming distance", ylabel="Validation/test images")
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(-0.12, 1.08, "B", transform=axes[1].transAxes, fontsize=12, fontweight="bold")
    figure.suptitle("Clinic v2 post-hoc shortcut diagnostic", fontsize=14, fontweight="bold")
    figure.savefig(FIGURES_DIR / f"{args.output_prefix}_audit.png", dpi=600, bbox_inches="tight")
    figure.savefig(FIGURES_DIR / f"{args.output_prefix}_audit.pdf", bbox_inches="tight")
    plt.close(figure)
    print(json.dumps({"report": str(report_path.resolve()), "baseline": baseline, "dhash": report["cross_split_dhash"]}, indent=2))


if __name__ == "__main__":
    main()
