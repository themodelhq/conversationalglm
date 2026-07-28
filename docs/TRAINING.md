# Training Guide

## Prepare text data

Input JSONL records use either a `messages` array or `prompt` and `response` strings. Assistant turns are the supervised labels.

```bash
python -m training.validate examples/conversations.jsonl --task sft
python -m training.prepare --config config/data.yaml --input examples/conversations.jsonl
python -m training.train --config config/train_sft.yaml
```

`training.prepare` normalizes text, rejects malformed records, creates deterministic train/validation splits, shards training JSONL, and trains a byte-level BPE tokenizer.

## Resume and distributed runs

```bash
python -m training.train --config config/train_sft.yaml --resume checkpoints/glm-sft/step-1000
GPUS=8 DEEPSPEED_CONFIG=config/deepspeed_zero3.json ./scripts/train_distributed.sh config/train_sft.yaml
GPUS=8 FSDP=1 ./scripts/train_distributed.sh config/train_sft.yaml
```

The trainer records accelerator state at `save_steps`, restores optimizer and scheduler state, evaluates at `eval_steps`, retains the lowest validation loss model in `best/`, and stops after the configured validation patience.

## Modal training records

| Task | Required JSONL fields |
|---|---|
| ASR | `audio_path`, `transcript`, optional `language` |
| TTS | `audio_path`, `transcript` |
| Speech emotion | `audio_path`, `emotion` |
| Vision | `image_path` |
| Video denoising | `video_path` |
| Lip sync | `audio_path`, `mouth_landmarks` |
| Motion/gesture | `features` (256 floats), `target` (7 floats) |
| Emotion generation | `features` (256 floats), `target` (8 floats) |
| Memory | `features` (256 floats), `target` (256 floats) |

```bash
python -m training.train_asr --train datasets/asr.jsonl --output checkpoints/asr
python -m training.train_tts --train datasets/tts.jsonl --output checkpoints/tts
python -m training.train_video --train datasets/video.jsonl --output checkpoints/video
```

Each modal entry point accepts `--train`, `--validation`, `--output`, `--epochs`, `--batch-size`, and `--lr`.
