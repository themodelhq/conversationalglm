# Training Platform

The React training platform is in `frontend/`; its authenticated orchestration API is in `backend/main.py`; and `backend/training_service.py` owns isolated local training processes.

## Start the platform

```bash
source .venv/bin/activate
export GLM_JWT_SECRET="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:5173`, create an account with a 12-character-or-longer password, then enter the workspace. For a containerized stack, set `POSTGRES_PASSWORD`, `GLM_JWT_SECRET`, and `GLM_MODEL_DIR`, then run `docker compose up --build`; the web platform is served on port 5173.

## Platform workflow

1. Open **Data Studio** and select SFT, DPO, or a specialist modality: ASR, TTS, speech emotion recognition, emotion generation, vision, video, motion, lip synchronization, gesture, or memory.
2. Upload a UTF-8 JSONL dataset using the task-specific schema in `docs/TRAINING.md`.
3. Open **Training Lab**, choose the matching learning objective or specialist target, then select its governed dataset.
4. Set model scale, context window, precision, optimizer, checkpoints, validation cadence, and experiment tracker.
5. Choose a single process, DeepSpeed ZeRO-3, or FSDP topology.
6. For a multi-node run, set node count, machine rank, master address, and master port. Each participating machine must launch the same generated configuration with its unique rank and shared model/data storage.
7. Launch the run and observe preparation, tokenizer training, SFT or DPO, validation, checkpoints, and process output in **Run Observatory**.
8. Review completed candidates in **Evaluate & Export**, then run the explicit evaluation and export commands shown there.

## Orchestration artifacts

Every launch writes immutable run files under:

```text
artifacts/runs/<run-id>/model.yaml
artifacts/runs/<run-id>/prepare.yaml
artifacts/runs/<run-id>/train.yaml
artifacts/runs/<run-id>/orchestrate.yaml
artifacts/runs/<run-id>/data/
artifacts/tokenizers/<run-id>/
checkpoints/<run-id>/
logs/training/<run-id>.log
```

The service keeps user-uploaded source assets under `data/uploads/<user-id>/`. Use durable object storage or a shared persistent volume before operating across multiple API replicas or compute nodes.

## Security boundary

The platform only accepts JSONL source assets for SFT and DPO orchestration. Restrict account provisioning, put the API behind TLS, use a production PostgreSQL database, use persistent volumes with least privilege, scan datasets before upload, and never run the API process with unrestricted host privileges. The local subprocess backend is intended for a controlled training environment; production schedulers should adapt `TrainingManager._command` to submit jobs to Kubernetes, Slurm, or another authenticated compute scheduler.
