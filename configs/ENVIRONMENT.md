# Training environment

- Python: 3.12.14
- PyTorch: 2.11.0+cu128
- torchvision: 0.26.0+cu128
- CUDA runtime bundled with PyTorch: 12.8
- Verified device: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM
- CUDA tensor operation: verified successfully on 2026-09-22

Install `torch` and `torchvision` from the official CUDA 12.8 PyTorch wheel index, then install the remaining packages from `environment.lock.txt`. The lock records the resolved local environment used by the benchmark.

The `.venv` directory is intentionally excluded from Git.

Set `YOLO_CONFIG_DIR` to the repository-local `.ultralytics` directory before importing or running Ultralytics. This avoids dependence on a user-profile settings file and keeps generated settings inside the project boundary.
