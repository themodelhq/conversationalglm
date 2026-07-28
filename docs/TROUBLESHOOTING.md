# Troubleshooting Guide

| Symptom | Resolution |
|---|---|
| API reports `model_ready: false` | Set `GLM_MODEL_PATH` to an exported model directory containing `model.safetensors`, `config.json`, and tokenizer files. |
| CUDA out of memory | Reduce batch size or sequence length, increase gradient accumulation, use bf16, enable gradient checkpointing, use FSDP or DeepSpeed ZeRO-3, and reduce generation length. |
| Slow first generation | Model, ASR, TTS, and diffusion pipelines load lazily. Warm each endpoint after deployment. |
| DPO loss is unstable | Verify prompt alignment, remove malformed preference pairs, reduce learning rate, and lower beta. |
| CTC loss is infinite | Ensure transcript byte-token lengths are shorter than mel frame lengths and remove empty/transcription-corrupt records. |
| Web client cannot reach API | Set `GLM_CORS_ORIGINS` to the exact frontend origin and use matching HTTPS/WSS schemes. |
| FFmpeg failure | Install FFmpeg and confirm it is available in `PATH`; verify audio/video codecs with `ffprobe`. |
| PostgreSQL connection error | Use `postgresql+asyncpg://` URL syntax, validate network policy, credentials, and TLS settings. |
