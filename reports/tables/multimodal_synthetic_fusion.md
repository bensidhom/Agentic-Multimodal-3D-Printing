# Synthetic multimodal paired replay

Values are mean ± sample SD across source-model seeds on 2,000 simulated test events per seed at 30% defect prevalence. Vision and AE held-out scores were paired by independent class-conditional resampling. This is simulation, not paired physical validation.

| Policy | Balanced accuracy | Macro F1 | Sensitivity | Specificity | False-positive rate | Miss rate |
|---|---:|---:|---:|---:|---:|---:|
| Gradient-boosted late fusion | 0.995 ± 0.003 | 0.996 ± 0.003 | 0.990 ± 0.007 | 1.000 ± 0.001 | 0.000 ± 0.001 | 0.010 ± 0.007 |
| Logistic late fusion | 0.970 ± 0.021 | 0.971 ± 0.018 | 0.956 ± 0.037 | 0.984 ± 0.008 | 0.016 ± 0.008 | 0.044 ± 0.037 |
| AND | 0.885 ± 0.016 | 0.890 ± 0.029 | 0.824 ± 0.018 | 0.945 ± 0.045 | 0.055 ± 0.045 | 0.176 ± 0.018 |
| Vision only | 0.901 ± 0.078 | 0.854 ± 0.108 | 0.995 ± 0.002 | 0.808 ± 0.156 | 0.192 ± 0.156 | 0.005 ± 0.002 |
| AE only | 0.777 ± 0.004 | 0.736 ± 0.005 | 0.828 ± 0.019 | 0.726 ± 0.012 | 0.274 ± 0.012 | 0.172 ± 0.019 |
| OR | 0.794 ± 0.059 | 0.703 ± 0.077 | 0.998 ± 0.003 | 0.589 ± 0.118 | 0.411 ± 0.118 | 0.002 ± 0.003 |

Late-fusion thresholds were selected on the synthetic validation split only. A paired print-job experiment remains required.
