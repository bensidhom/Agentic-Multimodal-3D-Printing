# Reproducibility and evidence scope

## Available without downloading images or models

The committed metrics, row predictions, split manifests and epoch logs support inspection of the reported component comparisons. `python scripts/verify_evidence.py` recomputes macro F1 from saved AE test decisions, checks frozen split counts and visual split integrity, and verifies that all eight publication figures exist. Unit tests exercise the shared-image adapter and validation-threshold selection.

The archived AE feature table is included. Its fitted pipelines can be regenerated with the original lockfile environment or an equivalent installation. A new machine will have different timing; do not overwrite or reinterpret the published timing as a new measurement.

## Required for new visual fitting

Obtain the exact Clinic version-2 dataset export from the stated Roboflow project. Use your own authorized credential, held in `ROBOFLOW_API_KEY`, never a committed notebook key. The frozen-view materializer uses recorded image hashes and normalized boxes to recreate the same train/validation/test membership. A failed hash match is a provenance failure, not permission to substitute a different image.

The runtime manifest uses paths for the current clone, while committed split manifests remain immutable records. Pretrained weights may be fetched by the chosen framework. Trained weights are not bundled because the 48 archived visual checkpoints total about 631 MB. Original model hashes and training/evaluation configuration remain in the saved records where recorded.

## Historical and publication provenance

Upstream HEAD IDs in the README were retrieved on 13 October 2026. They are not claimed to be the original training revisions. Some earlier local run manifests have no Git revision because the working project had no commit. Preserve that absence instead of backfilling a later commit.

Publication export only redacts machine-specific absolute paths and preserves recorded numerical content. `export_inventory.json` lists the pre-export and post-export digests. Digests belong in reproducibility metadata, not journal narrative.

## Architecture is not measured system validation

Fusion is an association architecture, not a validated accuracy result. Separate image and AE collections cannot be relabeled as synchronized recordings. The synthetic-paired replay experiment is explicitly separate and is excluded from the paper's physical-validation claims.

The source application contains role delegation and registered tool wrappers. The new package does not claim measured manufacturing-agent success, AMD deployment acceptance, independent command authorization, isolated code execution, physical stopping time, verified AE source localization or improved part quality. Those endpoints need dedicated tests and, for physical claims, appropriate hardware and independently grouped recordings.
