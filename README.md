# Conversational GLM

Open-source multimodal conversational assistant stack for training, preference optimization, retrieval, memory, speech, vision, image synthesis, avatar video rendering, and API deployment.

## Quick start

```bash
./scripts/bootstrap.sh
cp .env.example .env
python -m training.prepare --config config/data.yaml --input examples/conversations.jsonl
python -m training.train --config config/train_sft.yaml
GLM_MODEL_PATH=checkpoints/glm-sft python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

## Components

| Component | Entry point |
|---|---|
| Decoder-only conversational GLM | `models/glm.py` |
| Supervised training and DPO | `training/train.py` |
| ASR, TTS, speech emotion | `audio/` |
| Vision encoding and image generation | `vision/` |
| Video, lip synchronization, gestures | `video/` |
| Long-term memory | `memory/` |
| Document retrieval | `rag/` |
| FastAPI REST and WebSocket server | `backend/main.py` |
| PyTorch, ONNX, TensorRT export | `export/` |

## Commands

```bash
make install
make prepare
make train
make api
make check
python -m training.launch --config config/train_sft.yaml --gpus 8 --deepspeed config/deepspeed_zero3.json
python -m export.onnx --model checkpoints/glm-sft --output artifacts/model.onnx
python -m evaluation.language --model checkpoints/glm-sft --data data/processed/validation.jsonl
```

## Security

Use a unique high-entropy `GLM_JWT_SECRET`; serve through TLS; use PostgreSQL in multi-instance deployment; limit document uploads to trusted content; review tool registrations before exposing them; and obtain all dataset, voice, image, and video rights before training or serving.

## License

Apache-2.0. See `LICENSE`.

## Training Platform

The training platform combines a new React control surface with authenticated dataset governance, compute visibility, model-size presets, distributed topology controls, automated preparation, live run logs, experiment persistence, artifact paths, and export/evaluation guidance.

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
cd frontend && npm ci && npm run dev
```

See `docs/TRAINING_PLATFORM.md`. For managed deployment, see `docs/RENDER_NETLIFY.md`.
