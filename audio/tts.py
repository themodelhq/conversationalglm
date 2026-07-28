from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

class SpeechSynthesizer:
    """Coqui XTTS multilingual synthesis with controllable language, speaker sample, speed, and emotion text conditioning."""
    def __init__(self, model_name: str="tts_models/multilingual/multi-dataset/xtts_v2", device: str | None=None):
        self.model_name=model_name; self.device=device or ("cuda" if torch.cuda.is_available() else "cpu"); self._tts=None
    def _load(self):
        if self._tts is None:
            from TTS.api import TTS
            self._tts=TTS(self.model_name,progress_bar=False).to(self.device)
        return self._tts
    def synthesize(self, text: str, output: str | Path, language: str="en", speaker_wav: str | None=None, emotion: str="neutral", speed: float=1.0) -> Path:
        if not text.strip(): raise ValueError("text must not be empty")
        if speed <= 0: raise ValueError("speed must be positive")
        style=f"[{emotion}] {text}" if emotion and emotion != "neutral" else text
        kwargs={"text":style,"language":language,"file_path":str(output),"speed":speed}
        if speaker_wav: kwargs["speaker_wav"]=speaker_wav
        self._load().tts_to_file(**kwargs)
        return Path(output)
