# Installation

## Supported runtime

Python 3.10–3.12, PyTorch 2.3 or later, Node.js 20 or later, CUDA 12.1 or later for NVIDIA GPU inference/training, FFmpeg for video processing, and PostgreSQL 16 for production persistence.

## Local installation

```bash
./scripts/bootstrap.sh
source .venv/bin/activate
cp .env.example .env
```

Install the web console:

```bash
cd frontend
npm install
npm run build
```

## Optional feature groups

```bash
python -m pip install -e '.[train]'
python -m pip install -e '.[video,speech]'
python -m pip install -e '.[export]'
```

The video group installs Diffusers and media codecs. The speech group installs Coqui TTS. Use CUDA-compatible PyTorch wheels before installing these groups when using GPUs.

## Configuration

Copy `.env.example` to `.env`, generate a secret with `python -c "import secrets; print(secrets.token_urlsafe(48))"`, then set `GLM_JWT_SECRET`. Set `GLM_MODEL_PATH` to a directory containing `config.json`, `model.safetensors`, and tokenizer files.
