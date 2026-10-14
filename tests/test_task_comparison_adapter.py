from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "prepare_yolo_task_comparison.py"


class TaskComparisonAdapterTest(unittest.TestCase):
    def test_preserves_splits_and_excludes_multiclass_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "val", "test"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir(parents=True)

            examples = {
                ("train", "single.jpg"): "0 0.5 0.5 0.2 0.2\n",
                ("train", "multi.jpg"): "0 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.2 0.2\n",
                ("val", "background.jpg"): None,
                ("test", "single.jpg"): "1 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n",
            }
            for (split, filename), label in examples.items():
                Image.new("RGB", (32, 24), "white").save(root / split / "images" / filename)
                if label is not None:
                    (root / split / "labels" / Path(filename).with_suffix(".txt")).write_text(
                        label, encoding="utf-8"
                    )

            data_yaml = root / "data.yaml"
            data_yaml.write_text(
                yaml.safe_dump(
                    {
                        "path": ".",
                        "train": "train/images",
                        "val": "val/images",
                        "test": "test/images",
                        "names": ["defect_a", "defect_b"],
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "comparison.csv"
            summary = root / "summary.json"
            derived = root / "classification"
            detection = root / "detection"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--data-yaml",
                    str(data_yaml),
                    "--output-manifest",
                    str(manifest),
                    "--output-summary",
                    str(summary),
                    "--classification-output",
                    str(derived),
                    "--detection-output",
                    str(detection),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            result = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(result["images"], 4)
            self.assertEqual(result["comparison_eligibility"]["single_label_defect"], 2)
            self.assertEqual(result["comparison_eligibility"]["single_label_background"], 1)
            self.assertEqual(result["comparison_eligibility"]["multilabel_only"], 1)
            self.assertTrue((derived / "train" / "defect_a" / "single.jpg").exists())
            self.assertTrue((derived / "val" / "background" / "background.jpg").exists())
            self.assertFalse((derived / "train" / "defect_a" / "multi.jpg").exists())
            normalized = (detection / "test" / "labels" / "single.txt").read_text(
                encoding="utf-8"
            )
            self.assertEqual(len(normalized.split()), 5)
            self.assertEqual(result["annotations"]["polygon_instances_converted_to_boxes"], 1)


if __name__ == "__main__":
    unittest.main()
