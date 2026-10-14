"""Recreate the committed image split; never silently substitute a new export."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def materialize(images, destination, manifest):
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    image_index = {}
    for source in images.rglob("*"):
        if source.is_file() and source.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image_index.setdefault(digest(source), source)
    missing = [r for r in rows if r["image_sha256"] not in image_index]
    if missing:
        raise ValueError(f"Exact frozen export required: {len(missing)} image hashes are missing")
    runtime = []
    for row in rows:
        source = image_index[row["image_sha256"]]
        # Keep the frozen filename even if a fresh export uses a different name.
        name = row["image_path"].replace("\\", "/").rsplit("/", 1)[-1]
        split = row["split"]
        detection = destination / "clinic_v2_purged_detection" / split / "images" / name
        label = detection.parent.parent / "labels" / (Path(name).stem + ".txt")
        classification = destination / "clinic_v2_purged_classification" / split / row["classification_target"] / name
        for target in (detection, classification):
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != row["image_sha256"]:
                raise ValueError("Refusing to replace a conflicting prepared image")
            shutil.copy2(source, target)
        label.parent.mkdir(parents=True, exist_ok=True)
        boxes = row["normalized_boxes"].replace("|", "\n").strip()
        label.write_text(boxes + ("\n" if boxes else ""), encoding="utf-8")
        result = dict(row, image_path=str(detection.resolve()), label_path=str(label.resolve()),
                      detection_image_path=str(detection.resolve()), detection_label_path=str(label.resolve()),
                      classification_path=str(classification.resolve()))
        runtime.append(result)
    data = destination / "clinic_v2_purged_detection/data.yaml"
    data.write_text("path: " + json.dumps(str(data.parent.resolve())) +
                    "\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: spaghettification\n  1: underextrusion\n", encoding="utf-8")
    return runtime

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/clinic_v2_purged_manifest.csv")
    parser.add_argument("--runtime-manifest", type=Path, default=ROOT / "artifacts/clinic_v2_runtime_manifest.csv")
    args = parser.parse_args()
    before = digest(args.manifest)
    rows = materialize(args.images, args.destination, args.manifest)
    args.runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.runtime_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert digest(args.manifest) == before
    print(json.dumps({"images":len(rows), "frozen_manifest_unchanged":True, "runtime_manifest":str(args.runtime_manifest)}, indent=2))

if __name__ == "__main__":
    main()
