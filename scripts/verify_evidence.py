"""Verify archived predictions and frozen splits without fitting new models."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def verify(root=ROOT):
    visual = read_csv(root / "data/manifests/clinic_v2_purged_manifest.csv")
    counts = Counter(r["split"] for r in visual)
    assert counts == {"train":728,"val":200,"test":219}, counts
    for key in ("source_group_id", "image_sha256"):
        splits = defaultdict(set)
        for row in visual:
            splits[row[key]].add(row["split"])
        assert not any(len(s) != 1 for s in splits.values()), key
    acoustic = read_csv(root / "runs/acoustic/ae_overflow_feature_benchmark/frozen_row_split.csv")
    assert Counter(r["split"] for r in acoustic) == {"train":504,"validation":168,"test":168}
    assert Counter(r["label"] for r in acoustic) == {"0":479,"1":361}
    grouped = defaultdict(list)
    for row in read_csv(root / "runs/acoustic/ae_overflow_feature_benchmark/test_predictions.csv"):
        grouped[(row["model"], int(row["seed"]))].append(row)
    # Read the original per-run metrics rather than a generated aggregation.
    metric_paths = sorted((root / "runs/acoustic/ae_overflow_feature_benchmark").glob("*/metrics.json"))
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    expected = {(r["model"], int(r["seed"])): r for r in metrics}
    assert len(metrics) == len(expected) == 15
    for key, rows in grouped.items():
        assert len(rows) == 168
        assert Counter(r["y_true"] for r in rows) == {"0":96,"1":72}
        f1s = []
        for label in ("0","1"):
            tp = sum(r["y_true"] == label and r["prediction"] == label for r in rows)
            fp = sum(r["y_true"] != label and r["prediction"] == label for r in rows)
            fn = sum(r["y_true"] == label and r["prediction"] != label for r in rows)
            f1s.append(2*tp/(2*tp+fp+fn))
        assert math.isclose(sum(f1s)/2, expected[key]["macro_f1"], abs_tol=1e-12), key
    assert len(grouped) == 15
    for index in range(1,9):
        assert (root / f"assets/figures/figure_{index:02d}.png").is_file()
    return {"visual_images":len(visual),"visual_splits":dict(counts),"cross_split_group_and_hash_leaks":0,
            "acoustic_rows":len(acoustic),"ae_prediction_runs_verified":len(grouped),"figures":8}

if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
