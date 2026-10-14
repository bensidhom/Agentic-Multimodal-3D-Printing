# Classifier architecture comparison: purged Clinic-v2 view

Values are mean ± sample SD across seeds 17, 29, and 43 on the same 219-image test set. This is a temporal-overlap sensitivity analysis, not independent-print validation.

| Model | Accuracy | Balanced accuracy | Macro F1 | Brier | ECE | Median latency | Checkpoint |
|---|---:|---:|---:|---:|---:|---:|---:|
| EfficientNet-B0 | 0.967 ± 0.025 | 0.918 ± 0.061 | 0.936 ± 0.049 | 0.055 ± 0.026 | 0.070 ± 0.006 | 12.7 ± 0.1 ms | 16.3 MB |
| YOLO11n-cls | 0.954 ± 0.028 | 0.884 ± 0.077 | 0.908 ± 0.063 | 0.081 ± 0.057 | 0.042 ± 0.029 | 9.5 ± 1.2 ms | 3.2 MB |
| MobileNetV3-Small | 0.944 ± 0.014 | 0.853 ± 0.036 | 0.884 ± 0.034 | 0.098 ± 0.033 | 0.055 ± 0.024 | 9.4 ± 0.2 ms | 6.2 MB |
| ResNet-18 | 0.935 ± 0.011 | 0.829 ± 0.027 | 0.861 ± 0.029 | 0.108 ± 0.026 | 0.072 ± 0.010 | 7.2 ± 0.2 ms | 44.8 MB |

Selection conclusion: EfficientNet-B0 has the highest mean macro F1. YOLO11n-cls is smaller and faster, while ResNet-18 is fastest but much larger. Report all seed-level values because optimization variance is material.
