# GPU Guide

Install a CUDA-compatible NVIDIA driver, the matching PyTorch CUDA wheel, and NVIDIA Container Toolkit for Docker. Validate with:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Use `bf16` on Ampere or newer GPUs where supported. Set `CUDA_VISIBLE_DEVICES` to reserve devices. For distributed runs, launch with `training.launch`, ensure equal GPU counts per node, a routable master address, synchronized clocks, and compatible NCCL networking. Export `NCCL_DEBUG=INFO` when diagnosing collectives.
