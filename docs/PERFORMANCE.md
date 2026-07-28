# Performance Optimization Guide

Use packed, length-bucketed records for large-scale training; retain the current simple JSONL loader for correctness and small-to-medium datasets. Profile before optimizing:

```bash
python -m training.profile --output logs/profile --steps 50
tensorboard --logdir logs/profile
```

Enable bf16, gradient checkpointing, Flash/SDPA-compatible PyTorch kernels, fused optimizers where validated, DeepSpeed ZeRO-3 or FSDP for parameter sharding, and tokenizer/data-worker caching. Tune batch size against GPU memory. For serving, cap concurrent generations, cap token budgets, batch compatible requests only after validating tail latency, keep models warm, and offload long-running image/video tasks to a queue-backed worker service.
