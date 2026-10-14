# Nano-detector architecture comparison

Values are mean ± sample SD across seeds 17, 29, and 43 on the same 219-image purged-view test set. This is an archived temporal-overlap sensitivity analysis, not independent-print validation.

| Detector | Accuracy | Balanced accuracy | Macro F1 | AP50 | AP50:95 | Median latency | Size | Threshold range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 0.968 ± 0.021 | 0.928 ± 0.056 | 0.940 ± 0.043 | 0.720 ± 0.013 | 0.421 ± 0.007 | 12.9 ± 1.3 ms | 5.5 MB | 0.14–0.34 |
| YOLO26n | 0.936 ± 0.024 | 0.856 ± 0.046 | 0.877 ± 0.049 | 0.675 ± 0.010 | 0.395 ± 0.004 | 13.0 ± 0.1 ms | 5.4 MB | 0.05–0.13 |

Thresholds were selected on validation data separately for every fitted seed. Image-level macro F1 and localization AP answer different questions and are retained together.
