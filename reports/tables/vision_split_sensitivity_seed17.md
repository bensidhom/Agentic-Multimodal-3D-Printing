# Clinic-v2 split sensitivity comparison (seed 17)

The original export is an optimistic development split. The purged view is a temporal-overlap sensitivity analysis, not independent-print validation.

| Split view | Task | n | Accuracy | Balanced accuracy | Macro F1 | AP50 | AP50:95 | Median latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original | classification | 93 | 1.000 | 1.000 | 1.000 | — | — | 9.8 ms |
| original | detection | 93 | 0.989 | 0.993 | 0.977 | 0.727 | 0.426 | 20.3 ms |
| purged chronological | classification | 219 | 0.936 | 0.833 | 0.866 | — | — | 9.5 ms |
| purged chronological | detection | 219 | 0.945 | 0.865 | 0.893 | 0.722 | 0.423 | 14.4 ms |

The purged detector confidence threshold (0.14) was selected on validation data only and frozen before test evaluation.

The two test sets differ in size and temporal construction. Their score difference is a sensitivity diagnostic, not a paired causal estimate.
