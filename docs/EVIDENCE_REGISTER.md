# Validation evidence and resource register

Last updated: 2026-10-02

## Latest manuscript and source index

The current revision is `journal_outputs/JIM_image_led_revision.docx` and its PDF. Its image-led figures, attached red toolpath, caption changes and preservation checks are indexed in `journal_work/jim_submission/revision_visual_callouts/REVISION_NOTES.md`. The earlier source URLs, asset provenance and evidence boundaries remain indexed in `journal_work/jim_submission/revision_deployment_visuals/INVESTIGATION_AND_SOURCES.md`. The author confirmed an AMD Ryzen AI Max+ 395 deployment machine with **64 GB unified memory**. NVIDIA remains the visual training and benchmark host. Qwen3.6-35B-A3B is selected as an LLM upgrade, with local acceptance testing still required. YOLO11n detection and the small AE MLP are selected from the unchanged recognition comparisons. The user explicitly retained the attached red 3D toolpath; seven illustrated figures now use shorter labels and direct annotation arrows, with details in captions and manuscript text. All eight tables and 56 reference entries are unchanged. Fusion is architectural only; older sections below describing synthetic replay figures are historical, not the current manuscript scope. Previous artifacts are preserved.

## Purpose

This file is the persistent index for the Journal of Intelligent Manufacturing manuscript. It records what evidence exists, where it is stored, what each source can legitimately support, known data-quality problems, external resources, and the remaining validation work. Update this file whenever data, models, figures, results, or source material change.

## Current decision

- The existing repositories contain enough material to validate and reproduce substantial parts of the software architecture, computer-vision development, acoustic-feature pipeline, database design, agent workflow, and simulated printer control.
- External image datasets can provide an independent computer-vision benchmark if their source prints are kept together during splitting and the external test set is never used for tuning.
- Existing waveform/feature tables can be reanalysed, but the acoustic-emission evidence is not yet sufficient for a defensible independent physical validation. A serious label conflict exists in the largest workbook.
- OctoPrint's Virtual Printer is suitable for API, serial-protocol, state-machine, command-logging, safety-gate, recovery, and fault-injection tests. It is not evidence for physical motion-stop latency, sensor performance, spatial localization, or manufacturing outcomes.
- Synthetic waves and images may be used for unit tests, stress tests, rare-fault injection, and training augmentation. They must not replace a real, untouched test set or be described as physical experimental evidence.
- A Raspberry Pi is not required for the no-hardware software-validation phase; a PC/container can act as the edge host. A physical printer, camera, and contact-AE sensor/DAQ are required to support the manuscript's physical sensing and manufacturing-benefit claims.

## Current manuscript artifact

- Word manuscript: `journal_outputs/JIM_submission_demo_complete_red_evidence.docx`
- Build source: `journal_work/jim_submission/manuscript.md`
- Document builder: `journal_work/jim_submission/build_jim_white.py`
- Final QA: exported through Microsoft Word and visually inspected page by page; 25 pages, 8 tables and 13 figures, with no clipping, overlap or unreadable figure text detected.
- Intentional status: 186 red evidence-marker runs remain. They identify independent physical validation, system timing, safety-gate, manufacturing-outcome and declaration material that must be supplied or verified before actual submission.
- Experiment provenance: per-run manifests and checkpoints remain under `validation_platform/runs/`; aggregate machine-readable metrics, row-level outputs, manuscript tables and figures remain under `validation_platform/reports/`.

## Local source repositories

### Computer vision and cyber-infrastructure

Root:

`journal_work/repo_sources/Novel_Cyber_Infrastructure_src/Novel-Cyber-Infrastructure-Enabling-AI-Driven-Analytics-and-Digital-Twins-for-Real-Time-3D-Printing-main`

Important resources:

- Training notebook: `training_codes/train_yolov5_object_detection_on_custom_data.ipynb`
- Operational camera sessions include:
  - `Unnamed11_20250815_124331`: 3,212 raw images and 3,162 processed images.
  - `Unnamed1111_20250818_140210`: 4 raw images and 3 processed images.
- Detection export: `database/dynamic/cv_detections.csv`
  - Current content: header only; no detection rows.
- The repository contains a trained `best.pt` model artifact.

Notebook evidence found:

- Training set: 1,032 images, including 168 background images.
- Validation set: 110 images, including 16 background images and 146 labeled instances.
- Historical configuration: custom YOLOv5s initialized from scratch, 416-pixel input, batch size 64, and 30 epochs.
- Supplied checkpoint SHA-256: `bc055868a7228e0c10e6f06bc0e9e0d359239769b5cec3975bd7d2777d426b13`.
- Reported validation results:
  - All classes: precision 0.653, recall 0.627, mAP@0.50 0.597, mAP@0.50:0.95 0.279.
  - Spaghettification: 83 instances, precision 0.617, recall 0.524, mAP@0.50 0.503, mAP@0.50:0.95 0.264.
  - Under-extrusion: 63 instances, precision 0.689, recall 0.730, mAP@0.50 0.691, mAP@0.50:0.95 0.294.
- The notebook supports two classes, not the four classes currently described in parts of the draft.
- No independent frozen test set was found in the notebook. These are development/validation metrics, not final generalization estimates.
- The operational images are useful for interface replay, inference testing, annotation, and qualitative examples, but the available folder structure does not establish independent print-job labels or synchronized physical ground truth.

Allowed use now:

- Reproduce the supplied validation result.
- Benchmark inference speed and software throughput.
- Create print/job manifests and annotate a subset if job provenance can be recovered.
- Use operational images for replay and qualitative failure analysis.
- Do not report the thousands of frames as thousands of independent experiments.

### Acoustic-emission edge repository

Root:

`journal_work/repo_sources/Acoustic_Emission_Edge_src/Acoustic-Emission-Monitoring-at-the-Edge-for-Real-Time-Artificial-Intelligence-Applications-main`

Important resources:

- `data/training_with_labels.csv`
  - 164 rows.
  - Six features: amplitude, duration, energy, RMS, rise time, and counts.
  - Label counts: class 0 = 95; class 1 = 69.
  - The notebook uses a random row-level 80/20 split (131 training and 33 validation rows), not a grouped print/job split.
  - Development validation accuracy is approximately 0.6061 after 300 epochs.
- `data/overflow.csv`
  - 840 rows.
  - Label counts: class 0 = 479; class 1 = 361.
  - No print/job/run identifier was found.
- `data/vallen_training_dataset.xlsx`, sheet `Sheet1`, range `A1:G27858`
  - 27,857 data rows.
  - Six features plus label; no null values found.
  - Label counts: class 0 = 7,011; class 1 = 6,929; class 2 = 6,988; class 3 = 6,929.
  - Critical integrity issue: all 6,929 feature vectors assigned to class 1 are exact duplicates of the 6,929 feature vectors assigned to class 3. The multiclass result is invalid until the original labeling/provenance is recovered and this conflict is resolved.
- The repository contains H5 and ONNX model artifacts.

Allowed use now:

- Verify preprocessing, feature calculation, model loading, conversion, and inference code.
- Re-run exploratory analyses after excluding unresolved conflicting rows.
- Use known waveforms/features for replay and software tests.
- Do not claim an independent AE defect-classification result from the current row-random split.
- Do not combine the contradictory class-1/class-3 rows into a four-class benchmark.

Needed before a defensible AE result:

- Recover the class dictionary and acquisition log.
- Add sensor, channel, threshold, filter, gain, sampling, print-job, specimen, event-window, and timestamp provenance.
- Split by independent print job or specimen, never by individual hit/burst.
- Hold out complete jobs for final testing.
- If provenance cannot be recovered, describe this dataset as development data only and collect a new physical test set.

### Multi-agent SCADA repository

Root:

`journal_work/repo_sources/MultiAgent_SCADA_src/MultiAgent_SCADA-main`

Important resource:

- `data.db`
  - `computer_vision`: 18 rows (`job_001` = 7; `unknown_job` = 11).
  - `time_series`: 74 rows (`job_001` = 5; `job_002` = 1; `unknown_job` = 68).

Interpretation:

- This is an implementation snapshot suitable for schema inspection, query tests, migration tests, tool-permission tests, and agent workflow replay.
- It is not a manufacturing evaluation dataset and must not be used to estimate detector accuracy or intervention benefit.

## External datasets and resources

All links were checked on 2026-09-22. Re-check version, license, class definitions, and download integrity before use.

### Recommended image datasets

1. [Hu FDM 3D Printing Defect Dataset](https://github.com/wengmhu/FDM-3D-Printing-Defect-Dataset)
   - 1,912 images.
   - Five defect classes: cracking, layer shift, off-platform, stringing, and warping.
   - MIT license in the repository.
   - Best first external benchmark for the current machine because it is manageable in size.
   - Domain/classes do not exactly match the local two-class model, so define a class-mapping protocol before evaluation.
   - Materialized locally on 2026-09-22 as a compact 1,280-pixel classification benchmark: 1,912 images, 41 class-specific temporal segments, 1,342 training images, 276 validation images and 294 test images. No temporal segment, augmentation group or exact duplicate crosses partitions.
   - The archive contains class-folder labels only and no bounding boxes. It supports classification, not object-detection mAP or localization claims.
   - The archive count for `Cracking` is 472, whereas the repository README states 427; retain the archive-derived count and report the discrepancy.

2. [Clinic 3D Printing Defect Detect on Roboflow Universe](https://universe.roboflow.com/clinic-project-peqdq/clinic-3d-printing-defect-detect)
   - This is the public Roboflow project referenced by the supplied YOLOv5 training notebook. The current project page advertises approximately 1.2k images and four classes: over-extrusion, spaghettification, stringing and under-extrusion.
   - The notebook specifically downloaded project version 2, whose captured `data.yaml` and validation output contain only two classes: spaghettification and under-extrusion. Do not assume the current four-class project version is identical to the historical version-2 training corpus.
   - License shown by Roboflow Universe: CC BY 4.0.
   - Use the same frozen source split for an object-detection track and for a derived image-level classification track. This enables a fair classification-versus-detection comparison on shared image/event endpoints.
   - Do not expose or reuse the plaintext Roboflow API credential embedded in the old notebook; use a public export or a newly issued user credential and rotate the exposed key.

3. [Caxton Additive Manufacturing dataset](https://doi.org/10.17863/CAM.84082)
   - 1,272,273 labeled extrusion images from 192 printed parts.
   - Includes process labels such as flow, speed, Z offset, temperatures, timestamps, and nozzle X/Y.
   - Strong option for independent process-state testing and coordinate/context experiments.
   - Split by printed part/source run; never randomly split individual frames across train and test.

4. [High-resolution FFF defect dataset on Zenodo](https://zenodo.org/records/18628777)
   - Approximately 48.4 GB.
   - Includes extrusion errors, layer shifts, stringing, defect-free examples, and three camera angles.
   - Current free disk space is insufficient for a safe full local copy; use selective/streamed access or external storage.
   - License was not clearly populated on the record page during review, so usage rights must be verified before download or publication.

5. [MonolayerFFF](https://data.mendeley.com/datasets/k66f2gbgb4/3)
   - 776 surface images in three classes.
   - Useful as a small out-of-domain/transfer benchmark, subject to its license and exact class definitions.

### Acoustic and side-channel datasets

1. [SideChannel-3D](https://doi.org/10.5281/zenodo.13329934)
   - 144 audio and 144 vibration recordings, two printers, 12 objects, plus G-code/STL/video.
   - Useful for validating signal ingestion, windowing, replay, synchronization, and robustness.
   - It is microphone/audio/vibration data, not a matched contact-acoustic-emission dataset with the local Vallen features/classes. It cannot independently validate the local contact-AE classifier.

2. [LPBF acoustic-emission dataset](https://zenodo.org/records/13863838)
   - Different additive-manufacturing process and sensor domain.
   - May support method-development or transfer-learning experiments, but it is not a valid external FFF contact-AE test set.

Search conclusion:

- No closely matched, openly available FFF contact-AE dataset with the same sensor modality, six-feature schema, and defect labels was identified during this review.
- Therefore the strongest AE path is to recover provenance for the existing data and then collect a small new, grouped physical FFF test set.

### Simulation and documentation

- [OctoPrint Virtual Printer](https://docs.octoprint.org/en/main/development/virtual_printer.html)
  - Bundled development plugin for simulating serial communication without a printer.
  - Supports temperatures, heated bed, line/checksum behavior, resend scenarios, command buffers, emergency stop, artificial errors, timeouts, disconnects, throttling, sleeps, and serial logging.
  - Suitable for API/state/safety/fault-injection validation only.
- [OctoPrint printer REST API](https://docs.octoprint.org/en/main/api/printer.html)
- [Raspberry Pi 4 Model B documentation](https://pip.raspberrypi.com/categories/545-raspberry-pi-4-model-b)
- [Creality Ender-3 V2 support](https://www.creality.com/download/ender-3-v2-3d-printer-csco)
- [Logitech C270 documentation](https://www.logitech.com/assets/46735/hd-webcam-c270.pdf)
- [Vallen AMSY-6 specification](https://www.vallen.de/zdownload/pdf/AMSY-6_Spec.pdf)
- [smolagents documentation](https://huggingface.co/docs/smolagents/reference/agents)
- [SQLite limitations](https://www.sqlite.org/omitted.html)

## What can be completed without hardware

| Validation item | Existing/local data | External data | Virtual/synthetic | Defensible without hardware? |
|---|---:|---:|---:|---:|
| Repository provenance, hashes, manifests, environments | Yes | Not needed | Not needed | Yes |
| Vision code reproduction | Yes | Optional | Optional | Yes |
| Independent vision generalization benchmark | Limited | Yes | Augmentation only | Yes, if external real images are used correctly |
| Vision inference throughput/load tests | Yes | Yes | Yes | Yes |
| AE preprocessing/model-loading tests | Yes | Optional | Yes | Yes |
| Independent contact-AE defect performance | Insufficient | No matched source found | No | No |
| Database integrity, migrations, concurrency, permissions | Yes | Not needed | Yes | Yes |
| Agent benchmark and deterministic safety gate | Yes | Not needed | Yes | Yes |
| OctoPrint API, state machine, retries, reconnects, fault injection | Yes | Not needed | Yes | Yes |
| Software timing to API acknowledgement | Yes | Not needed | Yes | Yes, explicitly described as simulated |
| Camera/AE synchronization on a real print | Insufficient | No | No | No |
| Physical motion-stop latency | No | No | No | No |
| Physical localization error | No | No | No | No |
| Failed-print/material/completion benefit | No | No | No | No |

## Rules for using synthetic data

- Use synthetic images/waves for unit tests, controlled perturbations, sensitivity analysis, class balancing, augmentation, rare-event injection, and system stress testing.
- Generate synthetic data from a documented parameter range and save random seeds and generator versions.
- Keep synthetic samples out of the final real-world test set.
- Report results on real and synthetic data separately.
- Do not use the same source image, waveform, print, or augmented derivative in both training and testing.
- Do not call a virtual-printer response a physical printer result.

## Model-enhancement plan

The upgraded paper will compare models on identical frozen partitions rather than replacing the original model and reporting only the best run. Dataset provenance and split validity take priority over model novelty.

### Computer-vision model ladder

Train and compare:

1. Supplied YOLOv5 model/checkpoint as the historical thesis baseline.
2. YOLO11n as the mature modern real-time detector.
3. YOLO26n as the current edge-oriented detector candidate.
4. RT-DETR with a small backbone only as an optional architecture-diversity experiment if memory and training time permit.

Use pretrained weights, the same class definitions, the same job/group partitions, the same image resolution, equivalent augmentation, identical early-stopping rules and a documented compute budget. Run multiple seeds. Do not select the final model using the frozen test set.

Because the current GPU has 4 GB VRAM, begin with nano-scale models, 640-pixel inputs, automatic mixed precision and conservative batch sizes. Record training time, peak VRAM, model size, parameter count, FLOPs when supported, and CPU/GPU/ONNX inference latency in addition to detection accuracy. Check the Ultralytics AGPL-3.0/enterprise licensing terms before distributing a derivative application or trained deployment package.

Primary vision endpoints:

- Per-class precision, recall, F1, AP@0.50 and AP@0.50:0.95.
- Macro averages and event/job-level sensitivity.
- False alarms per print hour or per scenario.
- Reliability/calibration of confidence scores.
- External-dataset generalization without test-time tuning.
- Median, 95th-percentile and maximum inference time.

### Acoustic-emission model ladder

For the six engineered AE features, compare:

1. Regularized logistic regression as the interpretable baseline.
2. Random forest.
3. XGBoost or CatBoost as the principal nonlinear tabular candidate.
4. A small multilayer perceptron only as a secondary comparison.

Deep waveform models should be used only if genuine raw, time-ordered waveforms with trustworthy job/event labels are available. In that case compare MiniROCKET, a compact 1-D residual CNN and InceptionTime. Do not fabricate waveforms from the six summary features and then present a waveform classifier as independently validated.

All class-1/class-3 conflicting workbook rows remain quarantined. If independent print/job identifiers cannot be reconstructed, AE training is exploratory and its estimates must be labeled development results rather than generalization performance.

### Multimodal fusion ladder

Compare:

1. Vision only.
2. AE only.
3. Deterministic OR fusion.
4. Deterministic AND/windowed corroboration.
5. Calibrated logistic late fusion using detector scores, data freshness, temperature and printer state.
6. Gradient-boosted late fusion.
7. A temporal model such as a compact TCN only if enough independent labeled sequences exist.

Calibrate probabilities on a partition disjoint from model fitting. The safety decision remains a deterministic policy outside the learned fusion model.

### Training and selection protocol

- Create one immutable manifest before training.
- Group all frames, augmentations, bursts and derivatives from the same print/source in the same partition.
- Use training data for fitting, validation data for hyperparameters and thresholds, and the frozen test set once for final estimates.
- Run at least three documented seeds for stochastic models; increase this if variance is high.
- Select models using a preregistered utility criterion that balances job-level sensitivity, false pauses and latency rather than mAP alone.
- Export the selected candidates to ONNX and benchmark the exact exported artifact used by the demo.
- Save configuration, environment lock, seed, source commit, dataset hash, checkpoint hash, learning curves, predictions and failure cases for every run.

### Expected paper additions

- Model-comparison table covering accuracy, calibration, model size and latency.
- Per-class precision-recall curves.
- Confidence calibration/reliability diagram.
- Confusion matrices at frame level and event/job level.
- Accuracy-versus-latency Pareto plot.
- External-domain generalization table.
- Fusion ablation and safety-failure table.
- Qualitative failure cases with original images and predicted boxes.

## Recommended validation repository structure

Status: initialized on 2026-09-22 at `validation_platform/` with a Git `main` branch, evidence rules, the complete model matrix, benchmark policy and draft frozen experiment protocol.

The local repository contains:

```text
validation_platform/
  README.md
  CITATION.cff
  configs/
  data/
    manifests/
    external/
    interim/
    processed/
  experiments/
    vision_external/
    ae_audit/
    octoprint_virtual/
    agent_benchmark/
    database_load/
  src/
    vision/
    acoustic/
    control/
    agents/
    persistence/
  tests/
  reports/
    figures/
    tables/
    metrics/
  vendor/
    SOURCE_REPOSITORIES.md
```

Import the three source repositories with explicit commit/source provenance. Do not silently copy or modify source data. Store large external datasets outside Git and track manifests, hashes, licenses, and retrieval dates. Use DVC or Git LFS only after confirming storage availability.

## Minimum no-hardware experiment sequence

1. Freeze and hash all current datasets, models, notebooks, databases, and figures.
2. Build a row/file/job provenance manifest and identify which observations can be grouped into independent units.
3. Resolve or quarantine the AE class-1/class-3 conflict before any new training.
4. Reproduce the local two-class vision result from the notebook.
5. Select one manageable real external image dataset, define a written class mapping, and freeze an untouched external test set grouped by printed part/source run.
6. Run baseline, local-model, fine-tuned-model, and ablation comparisons with confidence intervals at the independent-unit level.
7. Add OctoPrint Virtual Printer tests for command authorization, stale data, wrong job identity, disconnect, timeout, resend, M112, invalid bounds, API failure, and logging.
8. Benchmark the deterministic controller, a single-agent baseline, and the multi-agent system on the same frozen tasks.
9. Run database integrity, concurrency, duplicate, orphan-file, and recovery tests.
10. Export machine-readable metrics, publication tables, high-resolution figures, environment lockfiles, seeds, and exact commands.

## Physical follow-up required for the full manuscript claim set

Minimum equipment:

- One FFF printer.
- One fixed camera/webcam with documented geometry and acquisition settings.
- One contact-AE sensor plus compatible preamplifier/DAQ, with documented coupling, gain, filters, threshold, sample rate, and calibration.
- The Raspberry Pi is optional; use it only if the paper specifically claims Raspberry Pi latency/utilization or deployment feasibility.

Run a power analysis before choosing the number of prints. Treat each independent print job/specimen as the experimental unit. Frames and AE hits are repeated observations inside a job. Use matched monitoring-disabled/monitoring-enabled trials, direct physical timing instrumentation, frozen models/thresholds, preregistered outcomes, and confidence intervals.

## Current compute/storage note

- Detected GPU: NVIDIA RTX 3050 Laptop GPU with 4 GB VRAM.
- This is adequate for inference and modest small-batch fine-tuning, but not for serving a 30B model locally without aggressive quantization/offloading or for comfortably storing the 48.4 GB Zenodo image dataset.
- Free space observed on the project drive during this audit: approximately 19.5 GB. Recheck before downloading data.

## Manuscript and figure outputs

- Current manuscript: `journal_outputs/JIM_submission_original_high_resolution_figures.docx`
- Editable manuscript source: `journal_work/jim_submission/manuscript.md`
- Figure-generation script: `journal_work/jim_submission/make_original_style_figures.mjs`
- High-resolution figure directory: `journal_work/jim_submission/figures/original_highres/`
- Latest rendered QA pages: `journal_work/jim_submission/render_original_highres/pages_final/`
- Hardware architecture figure: `journal_work/jim_submission/figures/original_highres/hardware_original_highres.png`

The manuscript deliberately marks missing empirical evidence in red. Do not remove a red marker until the corresponding result exists in a saved, auditable report and the value has been independently checked.

## Immediate recommendation

Proceed with a two-stage strategy:

1. Complete the no-hardware validation repository using the existing code/data, one manageable external vision dataset, OctoPrint Virtual Printer, deterministic/single-agent baselines, and explicit simulation labels.
2. Either acquire/borrow the minimum printer-camera-contact-AE rig for the physical claim set, or narrow the article to a cyber-infrastructure and simulated-safety paper and remove claims about physical sensor accuracy, physical stop latency, localization accuracy, and manufacturing benefit.

## Completed vision candidate benchmark (2026-09-22)

An authorized Roboflow project-version-2 export was downloaded without logging or committing the credential. The audited same-image corpus contains 1,235 images (1,032 train, 110 validation, 93 test), 1,658 defect instances, 192 background images, and the historical two-class label space (spaghettification and underextrusion). The audit found no malformed annotations or exact-hash cross-split duplicates. A later temporal audit established that the initial filename-family grouping was too fine-grained: every one of the three acquisition sequences (`test2`, `test5`, and `test6`) contributes frames to more than one original partition.

One seed-17 candidate comparison was completed on CUDA:0 using YOLO11n-cls and YOLO11n detection. The classifier achieved 1.000 test accuracy and macro F1 on 93 images. The detector, using confidence 0.25 selected on validation only, achieved 0.989 image-level accuracy, 0.977 macro F1, 0.727 AP50, and 0.426 AP50:95. It made one image-level error, mapping one spaghettification image to background. Median batch-one end-to-end latency was 9.8 ms for classification and 20.3 ms for detection; checkpoint sizes were 3.19 and 5.47 MB, respectively.

These values are optimistic development results, not final manuscript estimates. A completed shortcut/domain-cue audit found that 198 of 203 original validation/test images had a training image within four 64-bit difference-hash bits. The median blockwise SSIM between each evaluation image and its most similar training frame was 0.856; 20 pairs were at least 0.95. A fixed logistic-regression baseline using only global appearance features achieved 0.889 test macro F1. These diagnostics do not establish exact duplication and may partly encode genuine defect morphology, but they show that the original split cannot support strong independent-generalization claims. Detector score calibration remains unresolved; its current Brier/ECE values describe an uncalibrated decision-score normalization and must not be represented as calibrated probabilities.

### Purged chronological sensitivity view

A second, non-destructive view was built from the same source export to reduce adjacent-frame overlap. Within each acquisition-sequence/class stratum, the earliest 60% of source groups form training, the next 20% validation, and the final 20% test. Two source groups on both sides of each boundary are removed. This produced 728 training, 200 validation, and 219 test images; 88 images in 40 boundary groups were excluded. No exact hash or source group crosses a partition.

The stricter view materially reduced nearest-frame similarity: median blockwise SSIM fell from 0.856 to 0.565, pairs with SSIM at least 0.95 fell from 20/203 to 3/419, and the global-appearance baseline fell from 0.889 to 0.777 test macro F1. A seed-17 YOLO11n classifier reached 0.936 accuracy, 0.833 balanced accuracy, and 0.866 macro F1 on the 219-image test set. Fourteen of 28 background images were called underextrusion; the two defect classes otherwise had perfect recall in this split. A seed-17 YOLO11n detector used a confidence threshold of 0.14 selected on validation data only and reached 0.945 image-level accuracy, 0.865 balanced accuracy, 0.893 macro F1, 0.722 AP50, and 0.423 AP50:95. Its image-level errors were 11 background-to-underextrusion false positives and one underextrusion false negative.

This view is only a temporal-overlap sensitivity analysis. It is not independent-print validation: the dataset contains only three acquisition sequences, and class is partly confounded with sequence. The primary paper must state this limitation and should treat an independently collected print-level test set as future confirmatory work.

Four classifier architectures were subsequently completed under the same purged-view protocol using seeds 17, 29, and 43. EfficientNet-B0 had the highest mean test macro F1 (0.936 ± 0.049), followed by YOLO11n-cls (0.908 ± 0.063), MobileNetV3-Small (0.884 ± 0.034), and ResNet-18 (0.861 ± 0.029). Mean accuracy was 0.967, 0.954, 0.944, and 0.935, respectively. The best compact candidate is YOLO11n-cls at 3.2 MB and 9.5 ms median batch-one end-to-end latency; EfficientNet-B0 is the accuracy candidate at 16.3 MB and 12.7 ms. These are development/sensitivity results and not independent-print generalization estimates. Individual seed values must remain visible because optimization variance is material.

The YOLO11n detector was also completed for all three registered seeds. Mean image-level accuracy was 0.968 ± 0.021, balanced accuracy was 0.928 ± 0.056, and macro F1 was 0.940 ± 0.043. Mean localization performance was 0.720 ± 0.013 AP50 and 0.421 ± 0.007 AP50:95. Median batch-one end-to-end latency was 12.9 ± 1.3 ms and the checkpoint size was 5.5 MB. Its mean macro F1 was slightly above EfficientNet-B0 (0.940 versus 0.936), but the difference is small relative to seed variation and must not be framed as evidence of statistical superiority. The detector is preferred when defect location is operationally useful; YOLO11n-cls remains the smallest candidate, and EfficientNet-B0 remains the strongest classification-only candidate by mean macro F1.

YOLO26n detection was completed under the same protocol for all three seeds. It achieved mean image-level accuracy 0.936 ± 0.024, balanced accuracy 0.856 ± 0.046, macro F1 0.877 ± 0.049, AP50 0.675 ± 0.010 and AP50:95 0.395 ± 0.004, with 13.0 ± 0.1 ms batch-one end-to-end latency and a 5.4 MB checkpoint. YOLO11n therefore remains the selected localized detector because it was both more accurate and comparably compact under the registered comparison.

RT-DETR-L passed a one-epoch CUDA feasibility gate on the RTX 3050 Laptop GPU at 640 pixels and batch size 1. The run used approximately 1.47 GB peak reported training memory; the stripped checkpoint was 66.2 MB, the validated graph contained 31,987,850 parameters and 105.3 GFLOPs, and validation reported 29.4 ms inference per image. Its one-epoch validation AP50 of 0.545 and AP50:95 of 0.258 are pipeline-smoke outputs only and are excluded from the confirmatory ranking. A fair RT-DETR result would require the same registered training budget and all three seeds; that disproportionately expensive experiment was not run for the edge-oriented manuscript.

## Completed acoustic-feature development benchmark (2026-09-23)

The audited `overflow.csv` artifact contains 840 usable rows (479 normal and 361 defective), not 820. It has six engineered features and no verified print, specimen, acquisition-sequence or waveform identifiers. A frozen stratified row split used 504 training, 168 validation and 168 test rows; this supports only a development comparison. It does not support print-level confidence intervals, generalization claims or raw-waveform modeling.

Logistic regression, random forest, XGBoost, CatBoost and a small MLP were fitted with seeds 17, 29 and 43. Thresholds were selected on validation data separately for each fitted seed. The small MLP achieved the highest mean test macro F1 (0.773 ± 0.000) and balanced accuracy (0.781 ± 0.002), with defect sensitivity 0.833 ± 0.014 and specificity 0.729 ± 0.010. Corrected 2026-10-01 from the authoritative metrics: XGBoost had the highest mean AUROC (0.829 ± 0.003), followed by random forest (0.828 ± 0.002). Random forest's validation-selected operating point produced macro F1 0.755 ± 0.011 and substantially greater batch-one latency. These results must remain labeled row-level development sensitivity.

## Synthetic multimodal policy replay (2026-09-23)

The vision and AE archives are not physically paired. To exercise all planned fusion policies without inventing a paired experiment, held-out scores were resampled independently within true normal/defect strata. Each seed used 6,000 synthetic training, 2,000 validation and 2,000 test events at 30% defect prevalence. Gradient-boosted late fusion achieved synthetic macro F1 0.996 ± 0.003 and logistic late fusion 0.971 ± 0.018. OR reduced the synthetic miss rate to 0.002 ± 0.003 while increasing false positives to 0.411 ± 0.118; AND reduced false positives to 0.055 ± 0.045 while increasing misses to 0.176 ± 0.018.

These values are simulation outputs under a conditional-independence pairing assumption. They are useful for policy selection and integration testing only. They must not be used as evidence that multimodal fusion improves real print-job outcomes; the selected policies still require a paired physical evaluation on the same independent jobs.

Auditable outputs:

- `validation_platform/reports/metrics/vision_task_comparison_seed17.json`
- `validation_platform/reports/metrics/vision_task_comparison_seed17.csv`
- `validation_platform/reports/tables/vision_task_comparison_seed17.md`
- `validation_platform/reports/figures/vision_task_*_seed17.{png,pdf}`
- `validation_platform/runs/classification/clinic_v2_yolo11n_cls_seed17/`
- `validation_platform/runs/detection/clinic_v2_yolo11n_detect_seed17/`
- `validation_platform/reports/metrics/clinic_v2_shortcut_audit.json`
- `validation_platform/reports/metrics/clinic_v2_purged_shortcut_audit.json`
- `validation_platform/data/manifests/clinic_v2_purged_manifest.csv`
- `validation_platform/data/manifests/clinic_v2_purged_summary.json`
- `validation_platform/runs/classification/clinic_v2_purged_yolo11n_cls_seed17/test_evaluation/`
- `validation_platform/runs/detection/clinic_v2_purged_yolo11n_detect_seed17/`
- `validation_platform/reports/metrics/vision_split_sensitivity_seed17.{csv,json}`
- `validation_platform/reports/tables/vision_split_sensitivity_seed17.md`
- `validation_platform/reports/figures/vision_split_*_seed17.{png,pdf}`
- `validation_platform/reports/metrics/vision_classifier_multiseed.json`
- `validation_platform/reports/metrics/vision_classifier_multiseed_{rows,summary}.csv`
- `validation_platform/reports/tables/vision_classifier_multiseed.md`
- `validation_platform/reports/figures/vision_classifier_multiseed.{png,pdf}`
- `validation_platform/reports/metrics/vision_task_multiseed.json`
- `validation_platform/reports/metrics/vision_task_multiseed_{rows,summary}.csv`
- `validation_platform/reports/tables/vision_task_multiseed.md`
- `validation_platform/reports/figures/vision_task_multiseed.{png,pdf}`
- `validation_platform/runs/detection/clinic_v2_purged_yolo11n_detect_seed{17,29,43}/`
- `validation_platform/reports/metrics/vision_detector_architectures.json`
- `validation_platform/reports/metrics/vision_detector_architectures_{rows,summary}.csv`
- `validation_platform/reports/tables/vision_detector_architectures.md`
- `validation_platform/reports/figures/vision_detector_architectures.{png,pdf}`
- `validation_platform/runs/detection/clinic_v2_purged_yolo26n_detect_seed{17,29,43}/`
- `validation_platform/runs/detection/clinic_v2_purged_rtdetr_l_smoke_seed17/` (feasibility only; excluded from scientific ranking)
- `validation_platform/reports/metrics/ae_feature_benchmark.json`
- `validation_platform/reports/metrics/ae_feature_benchmark_rows.csv`
- `validation_platform/reports/tables/ae_feature_benchmark.md`
- `validation_platform/reports/figures/ae_feature_benchmark.{png,pdf}`
- `validation_platform/runs/acoustic/ae_overflow_feature_benchmark/`
- `validation_platform/reports/metrics/multimodal_synthetic_fusion.json`
- `validation_platform/reports/metrics/multimodal_synthetic_fusion_rows.csv`
- `validation_platform/reports/tables/multimodal_synthetic_fusion.md`
- `validation_platform/reports/figures/multimodal_synthetic_fusion.{png,pdf}`
- `validation_platform/runs/fusion/synthetic_paired_replay/`

## Clean submission manuscript and original-figure provenance (2026-09-30)

The clean Journal of Intelligent Manufacturing manuscript is stored at `journal_outputs/JIM_submission_clean_original_figures.docx`. Its generated text source is `journal_work/jim_submission/manuscript_clean_submission.md`, and the reproducible builder is `journal_work/jim_submission/build_jim_clean_submission.py`. The Word package contains five benchmark tables and nine original thesis/repository figures. Newly generated benchmark diagrams are not included.

Original figure mapping:

- Figure 1 system architecture: `journal_work/assets/image5.jpg`
- Figure 2 computer-vision workflow: `journal_work/assets/image8.jpg`
- Figure 3 acoustic-emission workflow: `journal_work/assets/image9.jpg`
- Figure 4 monitoring dashboard: `journal_work/assets/image10.png`
- Figure 5 original SQL schema (`computer_vision`, `time_series`, `Print_jobs`): `journal_work/assets/image11.jpg`
- Figure 6 supervisory-agent workflow: `journal_work/assets/image17.png`
- Figure 7 hardware and communications view: `journal_work/assets/image4.png`
- Figure 8 original computer-vision workflow and training trace: `journal_work/assets/image13.png`
- Figure 9 original acoustic-emission workflow and training traces: `journal_work/assets/image14.png`

The rendered 17-page verification copy is at `journal_work/jim_submission/render_clean/JIM_submission_clean_original_figures.pdf`; page images are in the same directory. Visual inspection confirmed that all nine figures, five tables, captions, references and the single red repository-location placeholder render without clipping. The fusion section remains explicitly labeled as a synthetic replay because the archived modalities are not physically paired.

## Expanded manuscript and independent evidence audit (2026-10-01)

The new revision is `journal_outputs/JIM_expanded_evidence_checked.docx`; the earlier document is preserved. The revision contains 11 figures, 7 tables and 46 references, with methods, results, discussion and limitations expanded. Original architecture, vision/AE workflows, dashboard, agent diagram and the exact old SQL image are retained. New white-background evaluation, comparison and diagnostic figures are generated from the saved records. All source paths, verification boundaries and submission requirements are documented in `journal_work/jim_submission/revision_oct01/VERIFICATION_AND_SUBMISSION_NOTES.md`.

Independent calculations reproduced classification scores from all 18 vision and 15 AE runs; AE AUROC, Brier and ECE were also checked. Aggregate mean/sample SD arithmetic and four transparent replay policies were independently recalculated. This does not rerun model inference or establish physical label validity. Critical corrections include the missing Print_jobs table/foreign keys in the saved database, the fusion replay's shared source-score pools, the XGBoost AUROC ranking, AE latency transcription errors, the passive VS600-Z1 sensor and the spotWave 201's documented 2-MHz specification. Earlier descriptions contradicted by these findings are superseded.

The document makes no claim that synthetic events are physical measurements. Independent-job generalization, synchronized fusion benefit, raw-AE provenance, physical pause timing and command-safety validation remain unverified. Author affiliations, contributions, funding, competing interests, permissions and the final repository release also require confirmation.
