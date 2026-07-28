from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

@dataclass
class GLMConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 4096
    dropout: float = 0.0
    vision_hidden_size: int = 768
    audio_hidden_size: int = 768
    num_emotions: int = 8
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    image_token_id: int = 4
    audio_token_id: int = 5
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GLMConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_file(cls, path: str | Path) -> "GLMConfig":
        path = Path(path)
        if path.suffix in {".yaml", ".yml"}:
            import yaml
            raw = yaml.safe_load(path.read_text())
            raw = raw.get("model", raw)
        else:
            raw = json.loads(path.read_text())
        return cls.from_dict(raw)

    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(asdict(self), indent=2))
