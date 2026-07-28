from __future__ import annotations
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import signal
import sys
from uuid import uuid4
import yaml
from sqlalchemy import select
from backend.settings import settings
from database.models import DatasetAsset, TrainingRun
from database.session import SessionLocal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_ROOT = Path(settings.storage_dir).expanduser().resolve()
SIZE_PRESETS = {
    "starter": {"hidden_size": 384, "intermediate_size": 1536, "num_hidden_layers": 6, "num_attention_heads": 6},
    "standard": {"hidden_size": 768, "intermediate_size": 3072, "num_hidden_layers": 12, "num_attention_heads": 12},
    "advanced": {"hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 24, "num_attention_heads": 16},
}

class TrainingManager:
    def __init__(self) -> None:
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.lock = asyncio.Lock()
        self.state_root = DEFAULT_STATE_ROOT
        self.using_ephemeral_storage = False

    def initialize_storage(self) -> Path:
        """Select a writable state directory before accepting uploads or launching jobs."""
        candidates = (DEFAULT_STATE_ROOT, Path("/tmp/conversational-glm-state"))
        failure: OSError | None = None
        for candidate in candidates:
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                probe = candidate / f".write-probe-{uuid4()}"
                probe.write_text("ok")
                probe.unlink()
                for child in ("uploads", "artifacts", "checkpoints", "logs"):
                    (candidate / child).mkdir(parents=True, exist_ok=True)
                self.state_root = candidate
                self.using_ephemeral_storage = candidate != DEFAULT_STATE_ROOT
                return candidate
            except OSError as error:
                failure = error
        raise RuntimeError(f"No writable state directory is available: {failure}")

    async def recover(self) -> None:
        async with SessionLocal() as session:
            rows = (await session.scalars(select(TrainingRun).where(TrainingRun.status.in_(["queued", "running", "stopping"]))))
            for run in rows:
                run.status = "interrupted"
                run.finished_at = datetime.now(timezone.utc)
            await session.commit()

    def _write_configs(self, run: TrainingRun, dataset: DatasetAsset) -> Path:
        run_dir = self.state_root / "artifacts" / "runs" / run.id
        data_dir = run_dir / "data"
        token_dir = self.state_root / "artifacts" / "tokenizers" / run.id
        output_dir = self.state_root / "checkpoints" / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        values = run.config_json
        model = {
            "model": {
                "name": "conversational-glm",
                "vocab_size": 32000,
                **SIZE_PRESETS[values["model_size"]],
                "max_position_embeddings": values["max_length"],
                "dropout": 0.0,
                "vision_hidden_size": 768,
                "audio_hidden_size": 768,
                "num_emotions": 8,
                "tie_word_embeddings": True,
                "gradient_checkpointing": True,
            }
        }
        prepare = {"data": {"sources": [dataset.path], "output_dir": str(data_dir), "tokenizer_dir": str(token_dir), "max_length": values["max_length"], "validation_ratio": 0.02, "seed": 42, "min_text_chars": 2, "languages": ["en", "yo", "ig", "ha", "fr", "es", "de", "pt", "zh", "ja", "ko", "ar"], "shard_size": 10000, "balance_by": "language"}}
        train = {"run": {"name": run.name, "output_dir": str(output_dir), "seed": 42, "resume_from": None}, "model": {"config": str(run_dir / "model.yaml"), "init_from": None}, "data": {"train": str(data_dir / "train.jsonl"), "validation": str(data_dir / "validation.jsonl"), "tokenizer": str(token_dir)}, "training": {"task": values["task"], "beta": 0.1, "epochs": values["epochs"], "max_steps": -1, "batch_size": values["batch_size"], "gradient_accumulation_steps": values["gradient_accumulation_steps"], "learning_rate": values["learning_rate"], "weight_decay": values["weight_decay"], "warmup_ratio": values["warmup_ratio"], "max_grad_norm": 1.0, "mixed_precision": values["mixed_precision"], "gradient_checkpointing": True, "save_steps": values["save_steps"], "eval_steps": values["eval_steps"], "logging_steps": 10, "early_stopping_patience": 8, "num_workers": 4, "deepspeed": "config/deepspeed_zero3.json" if values["strategy"] == "deepspeed" else None, "fsdp": values["strategy"] == "fsdp", "tracking": values["tracking"]}}
        model_path = run_dir / "model.yaml"; prepare_path = run_dir / "prepare.yaml"; train_path = run_dir / "train.yaml"; orchestrate_path = run_dir / "orchestrate.yaml"
        model_path.write_text(yaml.safe_dump(model, sort_keys=False)); prepare_path.write_text(yaml.safe_dump(prepare, sort_keys=False)); train_path.write_text(yaml.safe_dump(train, sort_keys=False)); orchestrate_path.write_text(yaml.safe_dump({"prepare_config": str(prepare_path), "train_config": str(train_path)}, sort_keys=False))
        return orchestrate_path

    def _launcher(self, values: dict, module: str, arguments: list[str]) -> list[str]:
        if values["gpu_count"] == 1 and values.get("node_count", 1) == 1 and values["strategy"] == "single":
            return [sys.executable, "-m", module, *arguments]
        command = ["accelerate", "launch", "--num_processes", str(values["gpu_count"]), "--num_machines", str(values.get("node_count", 1)), "--machine_rank", str(values.get("machine_rank", 0)), "--main_process_ip", str(values.get("main_process_ip", "127.0.0.1")), "--main_process_port", str(values.get("main_process_port", 29500))]
        if values["strategy"] == "deepspeed": command.extend(["--use_deepspeed", "--deepspeed_config_file", str(PROJECT_ROOT / "config" / "deepspeed_zero3.json")])
        if values["strategy"] == "fsdp": command.append("--use_fsdp")
        return [*command, "-m", module, *arguments]

    def _command(self, config_path: Path, values: dict, dataset_path: str, output_dir: str) -> list[str]:
        if values["task"] in {"sft", "dpo"}:
            return self._launcher(values, "training.orchestrate", ["--run-config", str(config_path)])
        return self._launcher(values, f"training.train_{values['task']}", ["--train", dataset_path, "--output", output_dir, "--epochs", str(values["epochs"]), "--batch-size", str(values["batch_size"]), "--lr", str(values["learning_rate"])])

    async def start(self, run_id: str) -> TrainingRun:
        async with self.lock:
            async with SessionLocal() as session:
                run = await session.get(TrainingRun, run_id)
                if run is None: raise ValueError("Training run not found")
                if run.status not in {"queued", "interrupted"}: raise ValueError(f"Run cannot start from state {run.status}")
                dataset = await session.get(DatasetAsset, run.dataset_id)
                if dataset is None or not Path(dataset.path).is_file(): raise ValueError("Training dataset is unavailable")
                config_path = self._write_configs(run, dataset)
                log_path = Path(run.log_path); log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("ab") as log:
                    command = self._command(config_path, run.config_json, dataset.path, run.output_dir)
                    log.write(("$ " + " ".join(command) + "\n").encode())
                    process = await asyncio.create_subprocess_exec(*command, cwd=str(PROJECT_ROOT), stdout=log, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
                run.status = "running"; run.process_id = process.pid; run.started_at = datetime.now(timezone.utc)
                await session.commit(); await session.refresh(run)
                self.processes[run.id] = process
                asyncio.create_task(self._watch(run.id, process))
                return run

    async def _watch(self, run_id: str, process: asyncio.subprocess.Process) -> None:
        code = await process.wait()
        self.processes.pop(run_id, None)
        async with SessionLocal() as session:
            run = await session.get(TrainingRun, run_id)
            if run is None: return
            summary = Path(run.output_dir) / "training_summary.json"
            metrics = json.loads(summary.read_text()) if code == 0 and summary.is_file() else {}
            run.metrics_json = metrics; run.return_code = code; run.finished_at = datetime.now(timezone.utc)
            run.status = "completed" if code == 0 else ("cancelled" if run.status == "stopping" else "failed")
            await session.commit()

    async def stop(self, run_id: str) -> bool:
        async with self.lock:
            async with SessionLocal() as session:
                run = await session.get(TrainingRun, run_id)
                if run is None or run.status not in {"queued", "running", "stopping"}: return False
                run.status = "stopping"; await session.commit()
                process = self.processes.get(run_id)
                try:
                    if process is not None: os.killpg(process.pid, signal.SIGTERM)
                    elif run.process_id is not None: os.kill(run.process_id, signal.SIGTERM)
                except ProcessLookupError: pass
                return True

    async def tail(self, run_id: str, limit: int = 300) -> tuple[str, list[str]]:
        async with SessionLocal() as session:
            run = await session.get(TrainingRun, run_id)
            if run is None: raise ValueError("Training run not found")
            path = Path(run.log_path)
            if not path.is_file(): return run.status, []
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return run.status, lines[-limit:]

training_manager = TrainingManager()
