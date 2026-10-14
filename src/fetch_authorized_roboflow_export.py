"""Fetch an authorized Roboflow export without persisting or logging its key.

Use the caller's ROBOFLOW_API_KEY environment variable. An explicit historical
notebook is supported for backwards compatibility but is not required or bundled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extended_path(path: Path) -> Path:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        resolved = "\\\\?\\" + resolved
    return Path(resolved)


def read_authorized_key(notebook: Path) -> str:
    payload = json.loads(extended_path(notebook).read_text(encoding="utf-8"))
    source = "\n".join(
        str(line)
        for cell in payload.get("cells", [])
        for line in cell.get("source", [])
        if cell.get("cell_type") == "code"
    )
    match = re.search(
        r"Roboflow\s*\(\s*api_key\s*=\s*['\"]([^'\"]+)['\"]",
        source,
    )
    if not match:
        raise RuntimeError("No Roboflow API key was found in the authorized notebook")
    key = match.group(1).strip()
    if len(key) < 12 or "*" in key or "YOUR" in key.upper():
        raise RuntimeError("The notebook contains a placeholder or malformed key")
    return key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, help="Optional legacy credential source; prefer ROBOFLOW_API_KEY")
    parser.add_argument("--workspace", default="clinic-project-peqdq")
    parser.add_argument("--project", default="clinic-3d-printing-defect-detect")
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--format", default="yolov5pytorch")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def request_export_link(
    session: requests.Session,
    key: str,
    workspace: str,
    project: str,
    version: int,
    export_format: str,
    timeout: int,
) -> str:
    endpoint = f"https://api.roboflow.com/{workspace}/{project}/{version}/{export_format}"
    deadline = time.monotonic() + timeout
    while True:
        try:
            response = session.get(
                endpoint,
                params={"api_key": key, "nocache": "true"},
                timeout=30,
            )
        except requests.RequestException as error:
            raise RuntimeError(f"Roboflow export request failed: {type(error).__name__}") from None
        if response.status_code not in (200, 202):
            raise RuntimeError(f"Roboflow export request returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError("Roboflow export response was not valid JSON") from None
        if response.status_code == 200 and payload.get("export", {}).get("link"):
            return str(payload["export"]["link"])
        if time.monotonic() >= deadline:
            raise TimeoutError("Roboflow export did not become ready before the timeout")
        progress = payload.get("progress")
        if progress is not None:
            print(f"Export preparation progress: {float(progress) * 100:.1f}%")
        time.sleep(2)


def download_archive(session: requests.Session, link: str, destination: Path) -> int:
    try:
        response = session.get(link, stream=True, timeout=(30, 180))
    except requests.RequestException as error:
        raise RuntimeError(f"Dataset download failed: {type(error).__name__}") from None
    if response.status_code != 200:
        raise RuntimeError(f"Dataset download returned HTTP {response.status_code}")
    total = 0
    with destination.open("wb") as stream:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                stream.write(chunk)
                total += len(chunk)
    return total


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Unsafe archive member rejected: {member.filename}")
        zipped.extractall(destination)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    metadata_path = args.metadata.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    archive = output.with_name(output.name + ".zip.partial")
    if partial.exists() or archive.exists():
        raise FileExistsError("A partial prior download exists; inspect it before retrying")

    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key and args.notebook is not None:
        key = read_authorized_key(args.notebook)
    if len(key) < 12:
        raise RuntimeError("Set your own authorized ROBOFLOW_API_KEY environment variable")
    session = requests.Session()
    session.headers["User-Agent"] = "validation-platform/1.0"
    try:
        link = request_export_link(
            session,
            key,
            args.workspace,
            args.project,
            args.version,
            args.format,
            args.timeout,
        )
        downloaded_bytes = download_archive(session, link, archive)
        archive_hash = sha256(archive)
        safe_extract(archive, partial)
        yaml_candidates = sorted(partial.rglob("data.yaml"))
        if len(yaml_candidates) != 1:
            raise RuntimeError(f"Expected one data.yaml, found {len(yaml_candidates)}")
        file_count = sum(path.is_file() for path in partial.rglob("*"))
        os.replace(partial, output)
        final_yaml = output / yaml_candidates[0].relative_to(partial)
        metadata = {
            "schema_version": 1,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "public_project_url": f"https://universe.roboflow.com/{args.workspace}/{args.project}",
                "workspace": args.workspace,
                "project": args.project,
                "version": args.version,
                "format": args.format,
            },
            "credential_handling": "Caller-authorized credential used at runtime; value not persisted or logged.",
            "archive": {
                "bytes": downloaded_bytes,
                "sha256": archive_hash,
                "deleted_after_verified_extraction": True,
            },
            "extraction": {
                "output": str(output),
                "file_count": file_count,
                "data_yaml": str(final_yaml),
                "data_yaml_sha256": sha256(final_yaml),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": "downloaded_and_extracted",
                    "version": args.version,
                    "format": args.format,
                    "bytes": downloaded_bytes,
                    "files": file_count,
                    "metadata": str(metadata_path),
                },
                indent=2,
            )
        )
    finally:
        key = ""
        if archive.exists():
            archive.unlink()
        if partial.exists():
            shutil.rmtree(partial)


if __name__ == "__main__":
    main()
