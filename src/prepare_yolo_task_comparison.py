"""Audit a YOLO detection dataset and derive a same-image classification view.

The detector keeps the original boxes. The classifier receives one image-level
class only when an image contains zero classes (background) or exactly one
unique box class. Images containing multiple defect classes are retained in the
audit manifest but are not silently forced into a single-label target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--classification-output", type=Path)
    parser.add_argument("--detection-output", type=Path)
    parser.add_argument("--copy", action="store_true", help="Copy instead of hard-linking derived images")
    return parser.parse_args()


def class_names(config: dict) -> list[str]:
    names = config.get("names")
    if isinstance(names, list):
        return [str(value) for value in names]
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    raise ValueError("YOLO data YAML must define names as a list or integer-keyed mapping")


def resolve_dataset_root(yaml_path: Path, config: dict) -> Path:
    root = Path(config.get("path", "."))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root


def resolve_images(root: Path, entry: str | list[str]) -> list[Path]:
    entries = [entry] if isinstance(entry, str) else entry
    images: list[Path] = []
    for value in entries:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
            # Roboflow YOLO exports commonly write ../train/images even when
            # data.yaml and train/ are extracted into the same directory. Its
            # SDK rewrites that path after download; reproduce that behavior
            # only when the literal relative path does not exist.
            if not candidate.exists():
                relative_parts = list(Path(value).parts)
                while relative_parts and relative_parts[0] == "..":
                    relative_parts.pop(0)
                rewritten = (root.joinpath(*relative_parts)).resolve()
                if rewritten.exists():
                    candidate = rewritten
        if candidate.is_file() and candidate.suffix.lower() == ".txt":
            for line in candidate.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                item = Path(line)
                if not item.is_absolute():
                    item = (candidate.parent / item).resolve()
                images.append(item)
        elif candidate.is_dir():
            images.extend(
                path for path in candidate.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif candidate.suffix.lower() in IMAGE_SUFFIXES:
            images.append(candidate)
        else:
            raise FileNotFoundError(f"Unresolved split entry: {candidate}")
    return sorted(set(path.resolve() for path in images))


def label_path(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def source_group_id(image: Path) -> str:
    """Collapse Roboflow augmentation hashes to their original source name."""
    return re.sub(r"\.rf\.[0-9a-f]+$", "", image.stem, flags=re.IGNORECASE)


def parse_labels(
    path: Path, class_count: int
) -> tuple[list[int], list[tuple[int, float, float, float, float]], list[str], list[str], bool]:
    if not path.exists():
        return [], [], [], [], False
    class_ids: list[int] = []
    normalized_boxes: list[tuple[int, float, float, float, float]] = []
    annotation_kinds: list[str] = []
    issues: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        try:
            class_id = int(float(fields[0]))
        except ValueError:
            issues.append(f"line_{line_number}_non_numeric")
            continue
        if not 0 <= class_id < class_count:
            issues.append(f"line_{line_number}_class_out_of_range")
            continue

        if len(fields) == 5:
            try:
                x, y, width, height = map(float, fields[1:])
            except ValueError:
                issues.append(f"line_{line_number}_non_numeric")
                continue
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                issues.append(f"line_{line_number}_invalid_normalized_box")
                continue
            annotation_kinds.append("box")
        elif len(fields) >= 7 and (len(fields) - 1) % 2 == 0:
            try:
                coordinates = list(map(float, fields[1:]))
            except ValueError:
                issues.append(f"line_{line_number}_non_numeric")
                continue
            if any(value < 0 or value > 1 for value in coordinates):
                issues.append(f"line_{line_number}_invalid_normalized_polygon")
                continue
            xs = coordinates[0::2]
            ys = coordinates[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            width, height = x_max - x_min, y_max - y_min
            if width <= 0 or height <= 0:
                issues.append(f"line_{line_number}_degenerate_polygon")
                continue
            x, y = (x_min + x_max) / 2, (y_min + y_max) / 2
            annotation_kinds.append("polygon")
        else:
            issues.append(f"line_{line_number}_unsupported_field_count_{len(fields)}")
            continue
        class_ids.append(class_id)
        normalized_boxes.append((class_id, x, y, width, height))
    return class_ids, normalized_boxes, annotation_kinds, issues, True


def materialize(source: Path, destination: Path, copy: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    yaml_path = args.data_yaml.resolve()
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    names = class_names(config)
    root = resolve_dataset_root(yaml_path, config)
    rows: list[dict[str, object]] = []

    for split in ("train", "val", "test"):
        entry = config.get(split)
        if entry is None:
            if split == "test":
                continue
            raise ValueError(f"Missing required split in data YAML: {split}")
        for image in resolve_images(root, entry):
            if not image.is_file():
                raise FileNotFoundError(image)
            labels = label_path(image)
            ids, normalized_boxes, annotation_kinds, issues, label_present = parse_labels(
                labels, len(names)
            )
            unique_ids = sorted(set(ids))
            if not unique_ids:
                target = "background"
                eligibility = "single_label_background"
            elif len(unique_ids) == 1:
                target = names[unique_ids[0]]
                eligibility = "single_label_defect"
            else:
                target = ""
                eligibility = "multilabel_only"
            with Image.open(image) as opened:
                width, height = opened.size
            rows.append(
                {
                    "split": split,
                    "image_path": str(image),
                    "image_sha256": sha256(image),
                    "source_group_id": source_group_id(image),
                    "width": width,
                    "height": height,
                    "label_path": str(labels),
                    "label_file_present": int(label_present),
                    "instance_count": len(ids),
                    "box_annotation_count": annotation_kinds.count("box"),
                    "polygon_annotation_count": annotation_kinds.count("polygon"),
                    "unique_class_count": len(unique_ids),
                    "class_ids": ";".join(map(str, unique_ids)),
                    "class_names": ";".join(names[index] for index in unique_ids),
                    "classification_target": target,
                    "comparison_eligibility": eligibility,
                    "annotation_issues": ";".join(issues),
                    "normalized_boxes": "|".join(
                        " ".join(
                            [str(box[0]), *(f"{value:.10f}" for value in box[1:])]
                        )
                        for box in normalized_boxes
                    ),
                }
            )

    hash_splits: dict[str, set[str]] = defaultdict(set)
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hash_splits[str(row["image_sha256"])].add(str(row["split"]))
        group_splits[str(row["source_group_id"])].add(str(row["split"]))
    cross_split_hashes = sorted(hash_value for hash_value, splits in hash_splits.items() if len(splits) > 1)
    cross_split_groups = sorted(group for group, splits in group_splits.items() if len(splits) > 1)

    if args.classification_output:
        destination_root = args.classification_output.resolve()
        for row in rows:
            if row["comparison_eligibility"] == "multilabel_only":
                continue
            source = Path(str(row["image_path"]))
            destination = destination_root / str(row["split"]) / str(row["classification_target"]) / source.name
            if destination.exists() and sha256(destination) != str(row["image_sha256"]):
                destination = destination.with_name(f"{source.stem}_{str(row['image_sha256'])[:10]}{source.suffix.lower()}")
            materialize(source, destination, args.copy)

    if args.detection_output:
        detection_root = args.detection_output.resolve()
        for row in rows:
            source = Path(str(row["image_path"]))
            image_destination = detection_root / str(row["split"]) / "images" / source.name
            if image_destination.exists() and sha256(image_destination) != str(row["image_sha256"]):
                image_destination = image_destination.with_name(
                    f"{source.stem}_{str(row['image_sha256'])[:10]}{source.suffix.lower()}"
                )
            materialize(source, image_destination, args.copy)
            label_destination = (
                detection_root
                / str(row["split"])
                / "labels"
                / image_destination.with_suffix(".txt").name
            )
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            label_text = str(row["normalized_boxes"]).replace("|", "\n")
            label_destination.write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")
        detection_yaml = {
            "path": str(detection_root),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(names),
            "names": names,
        }
        (detection_root / "data.yaml").write_text(
            yaml.safe_dump(detection_yaml, sort_keys=False), encoding="utf-8"
        )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    eligibility = Counter(str(row["comparison_eligibility"]) for row in rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    issue_rows = sum(bool(row["annotation_issues"]) for row in rows)
    box_annotations = sum(int(row["box_annotation_count"]) for row in rows)
    polygon_annotations = sum(int(row["polygon_annotation_count"]) for row in rows)
    summary = {
        "schema_version": 1,
        "source_yaml": str(yaml_path),
        "source_yaml_sha256": sha256(yaml_path),
        "dataset_root": str(root),
        "classes": names,
        "images": len(rows),
        "splits": dict(split_counts),
        "comparison_eligibility": dict(eligibility),
        "annotation_issue_rows": issue_rows,
        "annotations": {
            "box_instances": box_annotations,
            "polygon_instances_converted_to_boxes": polygon_annotations,
            "total_instances": box_annotations + polygon_annotations,
        },
        "cross_split_exact_duplicate_hashes": len(cross_split_hashes),
        "cross_split_hash_examples": cross_split_hashes[:20],
        "source_groups": len(group_splits),
        "cross_split_source_group_leaks": len(cross_split_groups),
        "cross_split_source_group_examples": cross_split_groups[:20],
        "classification_policy": "Preserve source split; map zero-box images to background, one-unique-class images to that class, and exclude multi-class images from single-label training.",
        "detection_policy": "Preserve source split and image bytes; retain YOLO boxes and convert normalized polygons to their tight axis-aligned bounding boxes.",
        "fair_comparison_rule": "Evaluate classification and detection on the intersection of eligible image IDs and report shared image/event endpoints; retain mAP only as a detection-specific endpoint.",
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
