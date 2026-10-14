"""Train an Ultralytics detector with a provenance sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PLATFORM_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PLATFORM_ROOT / ".ultralytics"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PLATFORM_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--project", type=Path, default=PLATFORM_ROOT / "runs" / "detection"
    )
    parser.add_argument("--name", default="yolo11n_detect")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data_yaml.resolve()
    dataset_manifest = args.dataset_manifest.resolve()
    if not data_yaml.is_file() or not dataset_manifest.is_file():
        raise FileNotFoundError("Detection data YAML or dataset manifest is missing")

    import torch
    import ultralytics
    from ultralytics import YOLO

    if str(args.device) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable")

    project = args.project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    run_dir = project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "pipeline_smoke_test" if args.smoke_test else "detection_benchmark",
        "scientific_use": (
            "Pipeline verification only; do not report as a paper result."
            if args.smoke_test
            else "Candidate benchmark; report only after the frozen protocol is satisfied."
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset": {
            "data_yaml": str(data_yaml),
            "data_yaml_sha256": sha256(data_yaml),
            "manifest": str(dataset_manifest),
            "manifest_sha256": sha256(dataset_manifest),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "git_revision": git_revision(),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    model = YOLO(args.model)
    try:
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            patience=args.patience,
            project=str(project),
            name=args.name,
            exist_ok=True,
            plots=not args.smoke_test,
            verbose=True,
        )
    except Exception as error:
        failed = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        failed["failed_utc"] = datetime.now(timezone.utc).isoformat()
        failed["failure"] = {"type": type(error).__name__, "message": str(error)[:1000]}
        (run_dir / "run_manifest.json").write_text(
            json.dumps(failed, indent=2), encoding="utf-8"
        )
        raise
    completed = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    completed["completed_utc"] = datetime.now(timezone.utc).isoformat()
    completed["best_checkpoint"] = str(run_dir / "weights" / "best.pt")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(completed, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
