# Classification versus object detection: three-seed purged-view comparison

Shared image-level values are mean ± sample SD across seeds 17, 29, and 43 on the same 219-image test set. This is a temporal-overlap sensitivity analysis, not independent-print validation.

| Model | Task | Accuracy | Balanced accuracy | Macro F1 | AP50 | AP50:95 | Median latency | Size |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n detector | object detection | 0.968 ± 0.021 | 0.928 ± 0.056 | 0.940 ± 0.043 | 0.720 ± 0.013 | 0.421 ± 0.007 | 12.9 ± 1.3 ms | 5.5 MB |
| EfficientNet-B0 | classification | 0.967 ± 0.025 | 0.918 ± 0.061 | 0.936 ± 0.049 | — | — | 12.7 ± 0.1 ms | 16.3 MB |
| YOLO11n-cls | classification | 0.954 ± 0.028 | 0.884 ± 0.077 | 0.908 ± 0.063 | — | — | 9.5 ± 1.2 ms | 3.2 MB |
| MobileNetV3-Small | classification | 0.944 ± 0.014 | 0.853 ± 0.036 | 0.884 ± 0.034 | — | — | 9.4 ± 0.2 ms | 6.2 MB |
| YOLO26n detector | object detection | 0.936 ± 0.024 | 0.856 ± 0.046 | 0.877 ± 0.049 | 0.675 ± 0.010 | 0.395 ± 0.004 | 13.0 ± 0.1 ms | 5.4 MB |
| ResNet-18 | classification | 0.935 ± 0.011 | 0.829 ± 0.027 | 0.861 ± 0.029 | — | — | 7.2 ± 0.2 ms | 44.8 MB |

YOLO11n detector is the strongest detector by mean shared-endpoint macro F1 and supplies localization. All detector thresholds were selected separately on validation data; localization AP and image-level macro F1 are retained as distinct endpoints.

Detector Brier score and ECE are not included because its threshold-normalized image-decision scores are not calibrated probabilities.
