# Vision task comparison (seed 17)

This is a same-image, single-seed benchmark. It is suitable as a candidate result and pipeline evidence, but final paper estimates require the registered additional seeds/models.

| Task | Accuracy | Balanced accuracy | Macro F1 | AP50 | AP50-95 | Median latency | P95 latency | Model size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classification | 1.000 | 1.000 | 1.000 | — | — | 9.8 ms | 10.4 ms | 3.19 MB |
| Object detection | 0.989 | 0.993 | 0.977 | 0.727 | 0.426 | 20.3 ms | 31.5 ms | 5.47 MB |

Detector image-level confidence was fixed at 0.25 on validation data. Detector decision scores are not calibrated probabilities; its Brier score/ECE are diagnostics and should not be presented as calibrated performance.

All source logs, predictions, checkpoints, arguments and hashes are indexed in `../metrics/vision_task_comparison_seed17.json`.
