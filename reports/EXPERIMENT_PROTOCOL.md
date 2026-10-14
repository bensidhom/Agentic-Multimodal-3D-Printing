# Frozen experiment protocol

Status: draft protocol; freeze before the first confirmatory training run.

## Research questions

1. Do modern vision detectors improve independent print-event detection over the supplied YOLOv5 baseline under an equivalent evaluation protocol?
2. Which model family best classifies the trustworthy engineered AE features without observation-level leakage?
3. When genuine raw waveforms are available, do waveform models outperform engineered-feature models at the independent print/job level?
4. Does multimodal fusion improve event sensitivity or reduce false pauses relative to the best single modality?
5. Which eligible model combination provides the best accuracy, calibration, latency, and model-size tradeoff for the simulated edge deployment?
6. On the same box-annotated images and frozen split, when is whole-image classification preferable to object detection, and what localization benefit and computational cost does detection add?

## Historical wording

Use:

> We evaluated successive generations of detection and classification models under a common frozen protocol to determine whether modern architectures improved the original prototype.

Do not use:

> The deployed system was continuously upgraded over time.

The second sentence implies a chronological deployment history that the available evidence does not establish.

## Fair-comparison rules

- Identical source/group partitions for every model in a modality.
- Equivalent training information and augmentation opportunities.
- Thresholds and calibration selected on validation data only.
- Detector confidence is selected on validation data by maximizing macro F1, then balanced accuracy, with the lowest threshold as the deterministic tie breaker.
- Frozen test data evaluated only after the model and threshold are fixed.
- Multiple seeds for stochastic training.
- External generalization reported separately from in-domain performance.
- Model selection includes false pauses and latency, not only frame-level accuracy.
- Every result must link to predictions, configuration, data manifest, environment, and checkpoint hash.
- For the classification-versus-detection comparison, derive image-level targets from the bounding-box annotations and preserve the detection dataset's source-group split.
- An image with boxes from exactly one defect class receives that image-level class; an image with no boxes receives `background`. Images containing boxes from multiple classes require a multi-label classifier and must not be forced into a single class. The primary single-label comparison uses the exact eligible-image intersection for both tasks.
- Convert detector outputs to image-level decisions using a validation-selected confidence rule. Compare both tasks on image/event sensitivity, specificity, false alarms, calibration and latency.
- Specificity and operational false-alarm rates are reported only when the frozen test set contains genuine normal/no-defect images. They are marked unavailable otherwise.
- Do not compare classification accuracy directly with detection mAP. Report classification-only and detection-only metrics in separate columns.

## Planned result tables

1. Dataset provenance and group-level partition table.
2. Vision model comparison.
3. AE engineered-feature model comparison.
4. Raw-waveform comparison, if the waveform gate is satisfied.
5. Calibration comparison.
6. Multimodal fusion ablation.
7. Accuracy-latency-model-size deployment comparison.
8. External generalization comparison.
9. Fault-injection and safety-control comparison.
10. Classification-versus-detection comparison on shared image/event endpoints.

## Planned figures

1. Dataset and experiment flow diagram.
2. Per-class precision-recall curves.
3. Frame-level and event-level confusion matrices.
4. Confidence reliability diagrams.
5. Accuracy-latency Pareto frontier.
6. AE feature importance and ablation.
7. Fusion event timeline examples.
8. Representative success and failure cases.
