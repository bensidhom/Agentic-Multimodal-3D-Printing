"""Create a non-destructive purged chronological Clinic v2 sensitivity split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data" / "manifests" / "clinic_v2_task_comparison_manifest.csv"
SOURCE_DETECTION = ROOT / "data" / "processed" / "clinic_v2_detection_boxes"
OUTPUT_MANIFEST = ROOT / "data" / "manifests" / "clinic_v2_purged_manifest.csv"
OUTPUT_SUMMARY = ROOT / "data" / "manifests" / "clinic_v2_purged_summary.json"
OUTPUT_CLASSIFICATION = ROOT / "data" / "processed" / "clinic_v2_purged_classification"
OUTPUT_DETECTION = ROOT / "data" / "processed" / "clinic_v2_purged_detection"
PATTERN = re.compile(r"^(.+)-(\d+)(?:-Copy)?_jpg$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--purge-groups", type=int, default=2)
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def assignment(rows: list[dict[str, str]], purge: int) -> tuple[dict[str, str], dict[str, str]]:
    groups: dict[str, dict[str, object]] = {}
    for row in rows:
        match = PATTERN.match(row["source_group_id"])
        if not match:
            raise ValueError(f"Unexpected source group: {row['source_group_id']}")
        group = groups.setdefault(
            row["source_group_id"],
            {
                "prefix": match.group(1).lower(),
                "frame": int(match.group(2)),
                "target": row["classification_target"],
            },
        )
        if group["target"] != row["classification_target"]:
            raise ValueError(f"Source group has conflicting targets: {row['source_group_id']}")
    strata: dict[tuple[str, str], list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for group_id, group in groups.items():
        strata[(str(group["prefix"]), str(group["target"]))].append((group_id, group))

    assigned: dict[str, str] = {}
    exclusions: dict[str, str] = {}
    for (prefix, target), items in strata.items():
        items.sort(key=lambda item: int(item[1]["frame"]))
        count = len(items)
        first = int(round(count * 0.60))
        second = int(round(count * 0.80))
        intervals = {
            "train": (0, max(0, first - purge)),
            "val": (min(count, first + purge), max(0, second - purge)),
            "test": (min(count, second + purge), count),
        }
        for split, (start, stop) in intervals.items():
            if stop <= start:
                raise ValueError(
                    f"Purge={purge} removes split {split} for {prefix}/{target}; reduce the purge."
                )
            for group_id, _ in items[start:stop]:
                assigned[group_id] = split
        for index, (group_id, _) in enumerate(items):
            if group_id not in assigned:
                boundary = "train_val" if first - purge <= index < first + purge else "val_test"
                exclusions[group_id] = f"purged_{boundary}_boundary"
    return assigned, exclusions


def main() -> None:
    args = parse_args()
    if args.purge_groups < 1:
        raise ValueError("purge-groups must be at least 1")
    with SOURCE_MANIFEST.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assigned, exclusions = assignment(rows, args.purge_groups)
    output_rows: list[dict[str, str]] = []
    counts: Counter[tuple[str, str]] = Counter()
    group_splits: dict[str, set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        group_id = row["source_group_id"]
        if group_id not in assigned:
            continue
        new_split = assigned[group_id]
        original_split = row["split"]
        image = Path(row["image_path"])
        image_name = image.name
        detection_image = SOURCE_DETECTION / original_split / "images" / image_name
        detection_label = SOURCE_DETECTION / original_split / "labels" / f"{image.stem}.txt"
        if not detection_image.is_file() or not detection_label.is_file():
            raise FileNotFoundError(f"Prepared detection source missing for {image_name}")

        classification_destination = (
            OUTPUT_CLASSIFICATION / new_split / row["classification_target"] / image_name
        )
        detection_image_destination = OUTPUT_DETECTION / new_split / "images" / image_name
        detection_label_destination = OUTPUT_DETECTION / new_split / "labels" / detection_label.name
        link_or_copy(image, classification_destination)
        link_or_copy(detection_image, detection_image_destination)
        link_or_copy(detection_label, detection_label_destination)

        updated = dict(row)
        updated["original_split"] = original_split
        updated["split"] = new_split
        updated["purge_policy"] = f"chronological_60_20_20_with_{args.purge_groups}_source_groups_each_boundary_side"
        updated["classification_path"] = str(classification_destination.resolve())
        updated["detection_image_path"] = str(detection_image_destination.resolve())
        updated["detection_label_path"] = str(detection_label_destination.resolve())
        output_rows.append(updated)
        counts[(new_split, row["classification_target"])] += 1
        group_splits[group_id].add(new_split)
        hash_splits[row["image_sha256"]].add(new_split)

    fieldnames = list(rows[0]) + [
        "original_split",
        "purge_policy",
        "classification_path",
        "detection_image_path",
        "detection_label_path",
    ]
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    yaml_text = (
        f"path: {OUTPUT_DETECTION.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
        "  0: spaghettification\n"
        "  1: underextrusion\n"
    )
    (OUTPUT_DETECTION / "data.yaml").write_text(yaml_text, encoding="utf-8")
    summary = {
        "schema_version": 1,
        "purpose": "purged_chronological_sensitivity_analysis_not_independent_print_validation",
        "purge_source_groups_each_boundary_side": args.purge_groups,
        "split_policy": "Within each acquisition-sequence/class stratum, retain the first 60% for train, next 20% for validation and last 20% for test, excluding source groups on both sides of each boundary.",
        "source_sequences": ["test2", "test5", "test6"],
        "independent_print_jobs_verified": False,
        "images_included": len(output_rows),
        "images_excluded_by_purge": len(rows) - len(output_rows),
        "source_groups_included": len(assigned),
        "source_groups_excluded_by_purge": len(exclusions),
        "counts_by_split_and_class": {
            split: {
                target: counts[(split, target)]
                for target in ("background", "spaghettification", "underextrusion")
            }
            for split in ("train", "val", "test")
        },
        "cross_split_source_group_leaks": sum(len(values) > 1 for values in group_splits.values()),
        "cross_split_exact_hash_leaks": sum(len(values) > 1 for values in hash_splits.values()),
        "manifest": str(OUTPUT_MANIFEST.resolve()),
        "classification_root": str(OUTPUT_CLASSIFICATION.resolve()),
        "detection_yaml": str((OUTPUT_DETECTION / "data.yaml").resolve()),
        "limitation": "All images originate from three acquisition sequences and class is partly confounded with sequence. This split is only a temporal-overlap sensitivity analysis.",
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
