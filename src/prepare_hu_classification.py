"""Materialize a compact, group-aware Hu FDM classification benchmark.

The public archive contains class-folder labels but no bounding boxes. This
script therefore creates a classification benchmark only. It keeps augmented
variants and temporally adjacent captures together and freezes a deterministic
train/validation/test assignment at the temporal-segment level.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PLATFORM_ROOT.parent
DEFAULT_ARCHIVE = PLATFORM_ROOT / "data" / "external" / "hu_fdm_dataset_download"
DEFAULT_OUTPUT = PLATFORM_ROOT / "data" / "processed" / "hu_fdm_classification_1280"
DEFAULT_MANIFEST = PLATFORM_ROOT / "data" / "manifests" / "hu_fdm_classification_manifest.csv"
DEFAULT_SUMMARY = PLATFORM_ROOT / "data" / "manifests" / "hu_fdm_classification_summary.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TIMESTAMP_PATTERN = re.compile(r"(\d{8})(\d{9})")


@dataclass(frozen=True)
class ArchiveImage:
    archive_path: str
    class_name: str
    filename: str
    base_capture: str
    variant: str
    timestamp: datetime
    archive_crc32: str
    archive_bytes: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def relative_workspace(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()


def parse_timestamp(stem: str) -> datetime:
    match = TIMESTAMP_PATTERN.search(stem)
    if not match:
        raise ValueError(f"No capture timestamp in {stem}")
    return datetime.strptime("".join(match.groups()) + "000", "%Y%m%d%H%M%S%f")


def list_archive_images(archive: Path) -> list[ArchiveImage]:
    images: list[ArchiveImage] = []
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            path = PurePosixPath(info.filename)
            if info.is_dir() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
                raise ValueError(f"Unsafe or unexpected archive path: {info.filename}")
            class_name = path.parts[-2]
            stem = path.stem
            if stem.endswith("_aug"):
                variant = "augmented"
            elif stem.endswith("_original"):
                variant = "original_variant"
            else:
                variant = "base"
            base_capture = re.sub(r"_(aug|original)$", "", stem)
            images.append(
                ArchiveImage(
                    archive_path=info.filename,
                    class_name=class_name,
                    filename=path.name,
                    base_capture=base_capture,
                    variant=variant,
                    timestamp=parse_timestamp(stem),
                    archive_crc32=f"{info.CRC:08x}",
                    archive_bytes=info.file_size,
                )
            )
    return sorted(images, key=lambda item: (item.class_name, item.timestamp, item.filename))


def assign_temporal_segments(images: list[ArchiveImage], gap_seconds: int) -> dict[tuple[str, str], str]:
    captures_by_class: dict[str, dict[str, datetime]] = defaultdict(dict)
    for image in images:
        captures_by_class[image.class_name][image.base_capture] = image.timestamp

    assignments: dict[tuple[str, str], str] = {}
    for class_name, captures in sorted(captures_by_class.items()):
        previous: datetime | None = None
        segment_index = -1
        for base_capture, timestamp in sorted(captures.items(), key=lambda item: (item[1], item[0])):
            if previous is None or (timestamp - previous).total_seconds() > gap_seconds:
                segment_index += 1
            segment_id = f"{class_name.lower()}_segment_{segment_index:02d}"
            assignments[(class_name, base_capture)] = segment_id
            previous = timestamp
    return assignments


def closest_subset(group_sizes: dict[str, int], target: float, max_selected: int) -> tuple[str, ...]:
    groups = sorted(group_sizes)
    choices: list[tuple[float, int, tuple[str, ...]]] = []
    for count in range(1, min(max_selected, len(groups)) + 1):
        for subset in itertools.combinations(groups, count):
            total = sum(group_sizes[group] for group in subset)
            choices.append((abs(total - target), count, subset))
    if not choices:
        raise ValueError("No eligible group subset")
    return min(choices)[2]


def freeze_splits(images: list[ArchiveImage], segments: dict[tuple[str, str], str]) -> dict[str, str]:
    image_counts_by_class_segment: dict[str, Counter[str]] = defaultdict(Counter)
    for image in images:
        segment = segments[(image.class_name, image.base_capture)]
        image_counts_by_class_segment[image.class_name][segment] += 1

    split_by_segment: dict[str, str] = {}
    for class_name, counter in sorted(image_counts_by_class_segment.items()):
        if len(counter) < 3:
            raise ValueError(f"Class {class_name} has fewer than three temporal segments")
        total = sum(counter.values())
        test_groups = closest_subset(dict(counter), target=0.15 * total, max_selected=len(counter) - 2)
        remaining = {group: size for group, size in counter.items() if group not in test_groups}
        val_groups = closest_subset(remaining, target=0.15 * total, max_selected=len(remaining) - 1)
        for group in counter:
            if group in test_groups:
                split_by_segment[group] = "test"
            elif group in val_groups:
                split_by_segment[group] = "val"
            else:
                split_by_segment[group] = "train"
    return split_by_segment


def reset_output_directory(output: Path) -> None:
    # The directory is generated exclusively by this script and is constrained
    # beneath validation_platform/data/processed.
    processed_root = (PLATFORM_ROOT / "data" / "processed").resolve()
    resolved = output.resolve()
    if processed_root not in resolved.parents:
        raise ValueError(f"Refusing to replace output outside {processed_root}: {resolved}")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def materialize(
    archive: Path,
    output: Path,
    images: list[ArchiveImage],
    segments: dict[tuple[str, str], str],
    splits: dict[str, str],
    max_edge: int,
    jpeg_quality: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    reset_output_directory(output)
    with zipfile.ZipFile(archive) as handle:
        for index, item in enumerate(images, start=1):
            segment = segments[(item.class_name, item.base_capture)]
            split = splits[segment]
            destination = output / split / item.class_name / f"{Path(item.filename).stem}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_bytes = handle.read(item.archive_path)
            if sha256_bytes(source_bytes) == "":
                raise AssertionError("Unreachable integrity guard")
            with Image.open(io.BytesIO(source_bytes)) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                original_width, original_height = image.size
                image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                processed_width, processed_height = image.size
                image.save(destination, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
            processed_hash = sha256_file(destination)
            record_id = hashlib.sha256(item.archive_path.encode("utf-8")).hexdigest()[:20]
            rows.append(
                {
                    "record_id": record_id,
                    "dataset_id": "hu_fdm_external",
                    "task": "image_classification",
                    "source_type": "external_recorded_image",
                    "class_name": item.class_name,
                    "source_group_id": segment,
                    "augmentation_group_id": f"{item.class_name}:{item.base_capture}",
                    "variant": item.variant,
                    "capture_timestamp": item.timestamp.isoformat(timespec="milliseconds"),
                    "partition": split,
                    "archive_path": item.archive_path,
                    "archive_crc32": item.archive_crc32,
                    "archive_bytes": item.archive_bytes,
                    "processed_path": relative_workspace(destination),
                    "processed_bytes": destination.stat().st_size,
                    "processed_sha256": processed_hash,
                    "original_width": original_width,
                    "original_height": original_height,
                    "processed_width": processed_width,
                    "processed_height": processed_height,
                    "annotation_type": "class_folder",
                    "bounding_boxes_available": 0,
                    "confirmatory_scope": "external_temporal_segment_classification",
                    "provenance_caveat": "temporal_segments_are_not_verified_independent_print_jobs",
                }
            )
            if index % 100 == 0 or index == len(images):
                print(f"processed {index}/{len(images)}")
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    split_by_source_group: dict[str, set[str]] = defaultdict(set)
    split_by_augmentation_group: dict[str, set[str]] = defaultdict(set)
    splits_by_hash: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["partition"])
        split_by_source_group[str(row["source_group_id"])].add(split)
        split_by_augmentation_group[str(row["augmentation_group_id"])].add(split)
        splits_by_hash[str(row["processed_sha256"])].add(split)
    leaking_source_groups = {key: sorted(value) for key, value in split_by_source_group.items() if len(value) > 1}
    leaking_augmentation_groups = {
        key: sorted(value) for key, value in split_by_augmentation_group.items() if len(value) > 1
    }
    duplicate_hashes_across_splits = {key: sorted(value) for key, value in splits_by_hash.items() if len(value) > 1}
    if leaking_source_groups or leaking_augmentation_groups or duplicate_hashes_across_splits:
        raise ValueError(
            json.dumps(
                {
                    "leaking_source_groups": leaking_source_groups,
                    "leaking_augmentation_groups": leaking_augmentation_groups,
                    "duplicate_hashes_across_splits": duplicate_hashes_across_splits,
                },
                indent=2,
            )
        )
    return {
        "source_group_count": len(split_by_source_group),
        "augmentation_group_count": len(split_by_augmentation_group),
        "unique_processed_hashes": len(splits_by_hash),
        "cross_split_source_group_leaks": 0,
        "cross_split_augmentation_group_leaks": 0,
        "cross_split_exact_duplicate_hashes": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--archive-sha256", default="")
    parser.add_argument("--gap-seconds", type=int, default=300)
    parser.add_argument("--max-edge", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    images = list_archive_images(args.archive)
    if len(images) != 1912:
        raise ValueError(f"Expected 1912 images from the repository description/archive, found {len(images)}")
    segments = assign_temporal_segments(images, args.gap_seconds)
    splits = freeze_splits(images, segments)
    rows = materialize(
        args.archive,
        args.output,
        images,
        segments,
        splits,
        args.max_edge,
        args.jpeg_quality,
    )
    validation = validate_rows(rows)
    write_manifest(args.manifest, rows)

    archive_hash = args.archive_sha256 or sha256_file(args.archive)
    manifest_hash = sha256_file(args.manifest)
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
        "source": {
            "repository": "https://github.com/wengmhu/FDM-3D-Printing-Defect-Dataset",
            "download_record": "https://drive.google.com/file/d/1kRmKws66yzoQFHBcoUd4sF6HLPnHlXE4/view",
            "archive_sha256": archive_hash,
            "archive_bytes": args.archive.stat().st_size,
            "license": "MIT as stated in the source repository; verify image-level rights before publication",
        },
        "task": "image_classification",
        "annotation_type": "class_folder_only",
        "bounding_boxes_available": False,
        "processing": {
            "max_edge": args.max_edge,
            "jpeg_quality": args.jpeg_quality,
            "output_root": relative_workspace(args.output),
        },
        "split_policy": {
            "temporal_gap_seconds": args.gap_seconds,
            "target_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "assignment_unit": "class_specific_temporal_segment",
            "variants_kept_together": True,
            "limitation": "temporal segments are not verified independent print jobs",
        },
        "counts": {
            "total": len(rows),
            "by_partition": dict(sorted(Counter(str(row["partition"]) for row in rows).items())),
            "by_class": dict(sorted(Counter(str(row["class_name"]) for row in rows).items())),
            "by_partition_and_class": {
                split: dict(
                    sorted(Counter(str(row["class_name"]) for row in rows if row["partition"] == split).items())
                )
                for split in ("train", "val", "test")
            },
            "by_variant": dict(sorted(Counter(str(row["variant"]) for row in rows).items())),
        },
        "integrity": validation,
        "manifest": {
            "path": relative_workspace(args.manifest),
            "sha256": manifest_hash,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
