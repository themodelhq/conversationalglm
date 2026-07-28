from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True)
    args = parser.parse_args()
    run_config_path = Path(args.run_config).resolve()
    settings = yaml.safe_load(run_config_path.read_text())
    prepare_config = Path(settings["prepare_config"]).resolve()
    train_config = Path(settings["train_config"]).resolve()
    run([sys.executable, "-m", "training.prepare", "--config", str(prepare_config)])
    run([sys.executable, "-m", "training.train", "--config", str(train_config)])


if __name__ == "__main__":
    main()
