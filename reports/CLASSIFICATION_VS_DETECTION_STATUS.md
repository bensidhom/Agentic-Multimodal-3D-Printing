# Classification versus object detection benchmark

Updated: 2026-09-22

## Decision

Run both tasks. Classification answers whether a frame contains a defect class;
object detection adds location and instance count. The comparison will use the
same source images and frozen partitions, then evaluate shared image-level
endpoints. Detection mAP remains a separate localization endpoint and will not
be compared directly with classification accuracy.

## Completed

- Audited the historical notebook. It trained a custom YOLOv5s from scratch at
  416 px, batch 64, for 30 epochs.
- Verified that the notebook's Roboflow project version 2 used two classes:
  spaghettification and under-extrusion. The current public project page lists
  four classes, so these label spaces must remain separate.
- Froze and leakage-checked the Hu et al. classification corpus: 1,912 images,
  1,342 train, 276 validation, and 294 test. This corpus has class-folder labels
  only and cannot support object-detection localization metrics.
- Verified the local CUDA training stack on an NVIDIA GeForce RTX 3050 Laptop
  GPU using a one-epoch YOLO11n-classification smoke test.
- Implemented provenance-recording training and evaluation runners.
- Implemented and tested a YOLO annotation auditor that creates a same-image,
  split-preserving classification view. Zero-box images become `background`,
  one-class images receive that class, and multi-class images are not silently
  forced into an invalid single class.
- Downloaded the authorized Roboflow project-version-2 YOLO export without
  logging or retaining the API credential. The archive contained 1,235 images
  (1,032 train, 110 validation, 93 test), 1,658 defect instances, 192
  zero-box/background images and two defect classes.
- Converted 75 polygon annotations to tight axis-aligned boxes and found no
  malformed annotations or exact-hash cross-split duplicates. The initial
  filename-family grouping was subsequently shown to be too fine-grained: all
  three acquisition sequences (`test2`, `test5`, and `test6`) cross the
  original train/validation/test boundaries.
- Trained a same-image YOLO11n classifier and detector on CUDA:0 (NVIDIA RTX
  3050 Laptop GPU). The classifier stopped after 19 epochs under the registered
  patience rule; the detector completed 30 epochs and retained epoch 24.
- Selected the detector image-decision confidence (0.25) on validation data
  only, then evaluated both retained checkpoints on the 93-image test split.
- Generated plot-ready CSV/JSON, source hashes, a Markdown table, and three
  publication figures in both 600-dpi PNG and vector PDF formats.

## Smoke-test result (not a manuscript result)

The one-epoch YOLO11n-classification run completed successfully. Its metrics are
stored under `runs/smoke/` and explicitly marked `pipeline_smoke_test`. They are
not tuned estimates and must not be copied into the paper. The completed
candidate benchmark below is stored separately under `runs/classification/`
and `runs/detection/`.

## Original-split development result (seed 17; optimistic)

| Endpoint | Classification | Object detection |
|---|---:|---:|
| Test images | 93 | 93 |
| Accuracy | 1.000 | 0.989 |
| Balanced accuracy | 1.000 | 0.993 |
| Macro F1 | 1.000 | 0.977 |
| AP50 | — | 0.727 |
| AP50:95 | — | 0.426 |
| Median end-to-end latency | 9.8 ms | 20.3 ms |
| P95 end-to-end latency | 10.4 ms | 31.5 ms |
| Checkpoint size | 3.19 MB | 5.47 MB |

The detector made one image-decision error: one spaghettification image was
classified as background. Its localization metrics were precision 0.603,
recall 0.723, AP50 0.727 and AP50:95 0.426. Underextrusion localized more
reliably (AP50:95 0.495) than spaghettification (0.358).

This is a single-seed development benchmark, not a final generalization
estimate. A completed shortcut audit found strong temporal/visual proximity
between the original partitions: 198 of 203 validation/test images had a
training image within four 64-bit difference-hash bits, the median nearest-pair
blockwise SSIM was 0.856, and a fixed global-appearance logistic baseline
reached 0.889 test macro F1. These findings do not prove exact duplication, but
they make the perfect classifier result unsuitable as primary evidence.

The detector's normalized image-decision scores are uncalibrated; its Brier
score and ECE are diagnostics rather than evidence of calibrated
probabilities.

## Purged chronological sensitivity analysis

Because only three acquisition sequences are available, an independent-print
split is impossible. A non-destructive alternative view was therefore created
without modifying the source export. Within each sequence/class stratum, the
first 60% of source groups are training, the next 20% validation, and the final
20% test; two source groups on both sides of every boundary are purged.

- Included: 1,147 images; excluded by purge: 88 images / 40 source groups.
- Train/validation/test: 728 / 200 / 219 images.
- Cross-split source-group leaks: 0; cross-split exact-hash leaks: 0.
- Nearest-pair median blockwise SSIM fell from 0.856 to 0.565; pairs at or
  above 0.95 SSIM fell from 20/203 to 3/419.
- The global-appearance baseline fell from 0.889 to 0.777 test macro F1.
- YOLO11n-cls seed 17 achieved 0.936 accuracy, 0.833 balanced accuracy, and
  0.866 macro F1 on the 219-image purged test set. Its 14 errors were all
  background frames predicted as underextrusion.
- YOLO11n detection seed 17, using confidence 0.14 selected on validation data
  only, achieved 0.945 image-level accuracy, 0.865 balanced accuracy, 0.893
  macro F1, 0.722 AP50, and 0.423 AP50:95. It correctly detected all 98
  spaghettification images at the image-decision level; 11 background frames
  became underextrusion false positives and one underextrusion frame became a
  false negative.

This remains a temporal-overlap sensitivity analysis, not independent-print
validation: class is partly confounded with sequence and all observations come
from only three videos. At the image-decision level detection slightly exceeded
classification in this single seed, while also providing localization; its
detector score normalization is not a calibrated probability.

## Three-seed classifier architecture comparison

All four registered classifiers were trained with seeds 17, 29, and 43 on the
purged view. Values below are test mean ± sample standard deviation; individual
seed rows are retained in the machine-readable report.

| Model | Accuracy | Balanced accuracy | Macro F1 | Median latency | Checkpoint |
|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.967 ± 0.025 | 0.918 ± 0.061 | 0.936 ± 0.049 | 12.7 ± 0.1 ms | 16.3 MB |
| YOLO11n-cls | 0.954 ± 0.028 | 0.884 ± 0.077 | 0.908 ± 0.063 | 9.5 ± 1.2 ms | 3.2 MB |
| MobileNetV3-Small | 0.944 ± 0.014 | 0.853 ± 0.036 | 0.884 ± 0.034 | 9.4 ± 0.2 ms | 6.2 MB |
| ResNet-18 | 0.935 ± 0.011 | 0.829 ± 0.027 | 0.861 ± 0.029 | 7.2 ± 0.2 ms | 44.8 MB |

EfficientNet-B0 is the accuracy candidate; YOLO11n-cls is the compact edge
candidate. The large seed spread—especially for EfficientNet-B0 and
YOLO11n-cls—must be shown rather than reporting only their best runs.

## Three-seed classification-versus-detection conclusion

YOLO11n detection completed the same seeds with its confidence threshold
selected independently on validation data for every run (0.14, 0.34 and 0.29).

| Model | Task | Accuracy | Balanced accuracy | Macro F1 | AP50 | AP50:95 | Median latency |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO11n detector | Detection | 0.968 ± 0.021 | 0.928 ± 0.056 | 0.940 ± 0.043 | 0.720 ± 0.013 | 0.421 ± 0.007 | 12.9 ± 1.3 ms |
| EfficientNet-B0 | Classification | 0.967 ± 0.025 | 0.918 ± 0.061 | 0.936 ± 0.049 | — | — | 12.7 ± 0.1 ms |
| YOLO11n-cls | Classification | 0.954 ± 0.028 | 0.884 ± 0.077 | 0.908 ± 0.063 | — | — | 9.5 ± 1.2 ms |

Detection slightly leads the best classifier on the shared endpoint and
supplies localization. The difference is small relative to seed variability;
the defensible conclusion is functional rather than statistical superiority:
use detection when the controller/operator needs location or instance count,
and use the compact classifier when a frame-level alert is sufficient.
Detector Brier/ECE values are excluded from the cross-task calibration
comparison because its threshold-normalized decision scores are not calibrated
probabilities.

## Plot and provenance sources

- Plot-ready metrics: `reports/metrics/vision_task_comparison_seed17.csv`
- Hash/provenance index: `reports/metrics/vision_task_comparison_seed17.json`
- Paper table: `reports/tables/vision_task_comparison_seed17.md`
- Publication figures: `reports/figures/vision_task_*_seed17.{png,pdf}`
- Classifier logs and predictions:
  `runs/classification/clinic_v2_yolo11n_cls_seed17/`
- Detector logs, validation threshold curve and predictions:
  `runs/detection/clinic_v2_yolo11n_detect_seed17/`
- Frozen dataset audit: `data/manifests/clinic_v2_task_comparison_summary.json`
- Original-split shortcut audit:
  `reports/metrics/clinic_v2_shortcut_audit.json`
- Purged split manifest and construction summary:
  `data/manifests/clinic_v2_purged_manifest.csv` and
  `data/manifests/clinic_v2_purged_summary.json`
- Purged shortcut audit:
  `reports/metrics/clinic_v2_purged_shortcut_audit.json`
- Purged classifier result:
  `runs/classification/clinic_v2_purged_yolo11n_cls_seed17/test_evaluation/`
- Purged detector result and frozen validation operating point:
  `runs/detection/clinic_v2_purged_yolo11n_detect_seed17/`
- Original-versus-purged plot-ready metrics and evidence hashes:
  `reports/metrics/vision_split_sensitivity_seed17.{csv,json}`
- Paper table:
  `reports/tables/vision_split_sensitivity_seed17.md`
- Publication figures (600-dpi PNG and vector PDF):
  `reports/figures/vision_split_*_seed17.{png,pdf}`
- Multi-seed classifier rows, summaries, source hashes and table:
  `reports/metrics/vision_classifier_multiseed_*`,
  `reports/metrics/vision_classifier_multiseed.json`, and
  `reports/tables/vision_classifier_multiseed.md`
- Multi-seed classifier publication figure (600-dpi PNG and vector PDF):
  `reports/figures/vision_classifier_multiseed.{png,pdf}`
- Multi-seed task-comparison rows, summaries, hashes and table:
  `reports/metrics/vision_task_multiseed*` and
  `reports/tables/vision_task_multiseed.md`
- Multi-seed task-comparison publication figure (600-dpi PNG and vector PDF):
  `reports/figures/vision_task_multiseed.{png,pdf}`

The run folders retain per-epoch `results.csv`, exact arguments, environment
and dataset hashes, `best.pt`/`last.pt`, generated PR/F1/confusion plots,
row-level predictions, group-bootstrap intervals, calibration diagnostics and
latency measurements. Smoke-test runs remain isolated under `runs/smoke/`.

## Primary result table

| Endpoint | Classification | Detection |
|---|---:|---:|
| Image-level macro F1 | Yes | Boxes converted to image decisions |
| Sensitivity | Yes | Yes |
| Specificity / false-positive rate | Only if background exists | Only if background exists |
| Calibration (Brier/ECE) | Yes | Validation-calibrated image confidence |
| Median and p95 batch-one latency | Yes | Yes |
| Model size | Yes | Yes |
| Balanced accuracy | Task-specific | — |
| AP50 / AP50:95 / localization IoU | — | Task-specific |

## Selection rule

Choose the least complex model that satisfies sensitivity, false-alarm,
calibration, and latency gates. Prefer detection only when localization or
instance counting materially improves the pause/control decision or operator
workflow. Report all eligible candidates and failure cases, not only the winner.
