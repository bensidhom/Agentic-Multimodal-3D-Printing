"""Inspect the Hu FDM archive without extracting it."""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath


def main() -> None:
    archive = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/external/hu_fdm_dataset_download")
    with zipfile.ZipFile(archive) as handle:
        image_names = [
            item.filename
            for item in handle.infolist()
            if PurePosixPath(item.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        non_images = [
            item.filename
            for item in handle.infolist()
            if not item.is_dir() and PurePosixPath(item.filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}
        ]

    classes = Counter(PurePosixPath(name).parts[-2] for name in image_names)
    variants = Counter()
    class_dates: dict[str, Counter[str]] = defaultdict(Counter)
    class_groups: dict[str, Counter[int]] = {}
    samples: dict[str, list[str]] = defaultdict(list)
    timestamps_by_class: dict[str, list[datetime]] = defaultdict(list)

    for name in image_names:
        path = PurePosixPath(name)
        class_name = path.parts[-2]
        stem = path.stem
        if stem.endswith("_aug"):
            variant = "augmented"
        elif stem.endswith("_original"):
            variant = "original_variant"
        else:
            variant = "base"
        variants[variant] += 1
        match = re.search(r"(\d{8})(\d{9})", stem)
        class_dates[class_name][match.group(1) if match else "unknown"] += 1
        if match and not stem.endswith(("_aug", "_original")):
            timestamps_by_class[class_name].append(datetime.strptime("".join(match.groups()) + "000", "%Y%m%d%H%M%S%f"))
        if len(samples[class_name]) < 8:
            samples[class_name].append(path.name)

    for class_name in sorted(classes):
        base_counts = Counter(
            re.sub(r"_(aug|original)$", "", PurePosixPath(name).stem)
            for name in image_names
            if PurePosixPath(name).parts[-2] == class_name
        )
        class_groups[class_name] = Counter(base_counts.values())

    segment_sizes_300: dict[str, list[int]] = {}
    for class_name, timestamps in sorted(timestamps_by_class.items()):
        ordered = sorted(set(timestamps))
        segments: list[list[datetime]] = []
        for timestamp in ordered:
            if not segments or (timestamp - segments[-1][-1]).total_seconds() > 300:
                segments.append([timestamp])
            else:
                segments[-1].append(timestamp)
        segment_sizes_300[class_name] = [len(segment) for segment in segments]

    report = {
        "archive": str(archive.resolve()),
        "image_count": len(image_names),
        "non_image_files": non_images,
        "classes": dict(sorted(classes.items())),
        "variants": dict(sorted(variants.items())),
        "dates_by_class": {key: dict(sorted(value.items())) for key, value in sorted(class_dates.items())},
        "augmentation_group_sizes_by_class": {
            key: dict(sorted(value.items())) for key, value in sorted(class_groups.items())
        },
        "sample_filenames": dict(sorted(samples.items())),
        "temporal_segments_by_gap_seconds": {
            str(gap_seconds): {
                class_name: 0
                if not timestamps
                else 1
                + sum(
                    1
                    for previous, current in zip(sorted(set(timestamps)), sorted(set(timestamps))[1:])
                    if (current - previous).total_seconds() > gap_seconds
                )
                for class_name, timestamps in sorted(timestamps_by_class.items())
            }
            for gap_seconds in (120, 300, 600, 1800)
        },
        "base_capture_counts_per_300_second_segment": segment_sizes_300,
        "annotation_assessment": "classification_folder_labels_only" if not non_images else "inspect_non_image_files",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
