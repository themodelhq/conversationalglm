# Hardware Guide

The reference configuration (`hidden_size=768`, 12 layers) is suitable for functional training and development. It is not a frontier-scale base model. Model quality, throughput, and realism scale with architecture size, data quality, and validated compute budgets.

| Workload | Minimum | Recommended |
|---|---:|---:|
| API, CPU-only | 8 CPU cores, 16 GB RAM | 16 CPU cores, 64 GB RAM |
| Reference SFT | 1 × 24 GB GPU | 4 × 80 GB GPUs |
| DPO | 1 × 48 GB GPU | 8 × 80 GB GPUs with ZeRO-3 |
| ASR/TTS | 1 × 16 GB GPU | 4 × 24 GB GPUs |
| Diffusion video | 1 × 24 GB GPU | 8 × 80 GB GPUs |
| Vector/document storage | 50 GB SSD | NVMe storage sized for corpus plus 30% overhead |

Use ECC memory for multi-day training, NVMe local scratch for shards and checkpoints, a high-bandwidth network for multi-node jobs, and independent model/data backups.
