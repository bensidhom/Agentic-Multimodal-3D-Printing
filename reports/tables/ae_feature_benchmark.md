# Acoustic-emission engineered-feature development benchmark

All values are mean ± sample SD across model seeds 17, 29, and 43 on one frozen 168-row test split. The 840-row source has no verified print/job identifiers, so this is development sensitivity only.

| Model | Balanced accuracy | Macro F1 | Defect sensitivity | Specificity | AUROC | Brier | ECE | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Small MLP | 0.781 ± 0.002 | 0.773 ± 0.000 | 0.833 ± 0.014 | 0.729 ± 0.010 | 0.814 ± 0.004 | 0.177 ± 0.004 | 0.127 ± 0.009 | 0.201 ± 0.005 ms |
| CatBoost | 0.763 ± 0.020 | 0.759 ± 0.014 | 0.773 ± 0.079 | 0.753 ± 0.039 | 0.816 ± 0.002 | 0.174 ± 0.002 | 0.129 ± 0.004 | 0.241 ± 0.004 ms |
| Logistic regression | 0.762 ± 0.000 | 0.755 ± 0.000 | 0.806 ± 0.000 | 0.719 ± 0.000 | 0.773 ± 0.000 | 0.189 ± 0.000 | 0.158 ± 0.000 | 0.187 ± 0.002 ms |
| Random forest | 0.762 ± 0.012 | 0.755 ± 0.011 | 0.806 ± 0.024 | 0.719 ± 0.000 | 0.828 ± 0.002 | 0.175 ± 0.002 | 0.130 ± 0.004 | 49.261 ± 0.526 ms |
| XGBoost | 0.750 ± 0.010 | 0.741 ± 0.016 | 0.806 ± 0.050 | 0.694 ± 0.069 | 0.829 ± 0.003 | 0.178 ± 0.001 | 0.126 ± 0.014 | 0.334 ± 0.027 ms |

Thresholds were selected on the validation split separately for each fitted seed. Test data were not used for model or threshold selection.

Independent grouped acquisition is still required before reporting print-level confidence intervals or manufacturing generalization.
