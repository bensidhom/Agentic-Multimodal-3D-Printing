"""Quantify whether Clinic v2 can support a purged temporal image split."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "reports" / "metrics" / "clinic_v2_shortcut_features.csv"
OUTPUT = ROOT / "reports" / "metrics" / "clinic_v2_split_feasibility.json"
PATTERN = re.compile(r"^(.+)-(\d+)(?:-Copy)?_jpg$", re.IGNORECASE)


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def load_groups() -> list[dict]:
    groups: dict[str, dict] = {}
    with FEATURES.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            match = PATTERN.match(row["source_group_id"])
            if not match:
                raise ValueError(f"Unexpected source group: {row['source_group_id']}")
            group = groups.setdefault(
                row["source_group_id"],
                {
                    "id": row["source_group_id"],
                    "prefix": match.group(1),
                    "frame": int(match.group(2)),
                    "target": row["target"],
                    "hashes": [],
                    "images": 0,
                },
            )
            group["hashes"].append(int(row["dhash_hex"], 16))
            group["images"] += 1
    return sorted(groups.values(), key=lambda row: (row["prefix"], row["frame"]))


def group_distance(left: dict, right: dict) -> int:
    return min((a ^ b).bit_count() for a in left["hashes"] for b in right["hashes"])


def chronological_assignment(groups: list[dict], purge: int) -> dict[str, str]:
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for group in groups:
        strata[(group["prefix"], group["target"])].append(group)
    assignment: dict[str, str] = {}
    for rows in strata.values():
        rows.sort(key=lambda row: row["frame"])
        count = len(rows)
        first = int(round(count * 0.60))
        second = int(round(count * 0.80))
        intervals = {
            "train": (0, max(0, first - purge)),
            "val": (min(count, first + purge), max(0, second - purge)),
            "test": (min(count, second + purge), count),
        }
        for split, (start, stop) in intervals.items():
            for row in rows[start:stop]:
                assignment[row["id"]] = split
    return assignment


def overlap_summary(groups: list[dict], assignment: dict[str, str], threshold: int = 4) -> dict:
    kept = [group for group in groups if group["id"] in assignment]
    minimum = 64
    pairs = 0
    for left_index, left in enumerate(kept):
        for right in kept[left_index + 1 :]:
            if assignment[left["id"]] == assignment[right["id"]]:
                continue
            distance = group_distance(left, right)
            minimum = min(minimum, distance)
            pairs += int(distance <= threshold)
    split_groups = Counter(assignment.values())
    split_images = Counter()
    split_classes: dict[str, Counter] = defaultdict(Counter)
    for group in kept:
        split = assignment[group["id"]]
        split_images[split] += group["images"]
        split_classes[split][group["target"]] += group["images"]
    return {
        "purge_source_groups_each_boundary_side": None,
        "kept_source_groups": len(kept),
        "excluded_source_groups": len(groups) - len(kept),
        "split_source_groups": dict(split_groups),
        "split_images": dict(split_images),
        "split_class_images": {key: dict(value) for key, value in split_classes.items()},
        "minimum_cross_split_dhash_distance": minimum,
        "cross_split_group_pairs_at_or_below_4_bits": pairs,
    }


def component_summary(groups: list[dict], threshold: int) -> dict:
    dsu = DisjointSet(len(groups))
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            if group_distance(groups[left], groups[right]) <= threshold:
                dsu.union(left, right)
    components: dict[int, list[dict]] = defaultdict(list)
    for index, group in enumerate(groups):
        components[dsu.find(index)].append(group)
    sizes = sorted((len(rows) for rows in components.values()), reverse=True)
    largest = sorted(components.values(), key=len, reverse=True)[:10]
    return {
        "threshold_bits": threshold,
        "components": len(components),
        "largest_component_source_groups": sizes[0],
        "components_with_multiple_classes": sum(
            len({row["target"] for row in rows}) > 1 for rows in components.values()
        ),
        "largest_components": [
            {
                "source_groups": len(rows),
                "images": sum(row["images"] for row in rows),
                "prefixes": dict(Counter(row["prefix"] for row in rows)),
                "classes": dict(Counter(row["target"] for row in rows)),
                "frame_min": min(row["frame"] for row in rows),
                "frame_max": max(row["frame"] for row in rows),
            }
            for rows in largest
        ],
    }


def main() -> None:
    groups = load_groups()
    purges = []
    for purge in range(0, 21):
        summary = overlap_summary(groups, chronological_assignment(groups, purge))
        summary["purge_source_groups_each_boundary_side"] = purge
        purges.append(summary)
    report = {
        "schema_version": 1,
        "source_groups": len(groups),
        "sequences": dict(Counter(row["prefix"] for row in groups)),
        "classes": dict(Counter(row["target"] for row in groups)),
        "chronological_60_20_20_purge_sweep": purges,
        "perceptual_components": [component_summary(groups, threshold) for threshold in (0, 1, 2, 4)],
        "decision_rule": "A development split is acceptable only if every split retains every class and no cross-split source-group pair has dHash distance <=4. This does not establish independent-print generalization.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = [
        {
            "purge": row["purge_source_groups_each_boundary_side"],
            "kept": row["kept_source_groups"],
            "min_distance": row["minimum_cross_split_dhash_distance"],
            "near_pairs": row["cross_split_group_pairs_at_or_below_4_bits"],
            "images": row["split_images"],
        }
        for row in purges
    ]
    print(json.dumps({"output": str(OUTPUT.resolve()), "purge_sweep": compact, "components": report["perceptual_components"]}, indent=2))


if __name__ == "__main__":
    main()
