from __future__ import annotations
from pathlib import Path
from typing import Any
import torch
from audio.features import load_audio

class SpeechRecognizer:
    def __init__(self, model_id: str="openai/whisper-small", device: str | None=None):
        self.model_id=model_id; self.device=device or ("cuda:0" if torch.cuda.is_available() else "cpu"); self._pipe=None
    def _load(self):
        if self._pipe is None:
            from transformers import pipeline
            kwargs={"model": self.model_id, "device": 0 if self.device.startswith("cuda") else -1}
            if self.device.startswith("cuda"): kwargs["torch_dtype"]=torch.float16
            self._pipe=pipeline("automatic-speech-recognition",**kwargs)
        return self._pipe
    def transcribe(self, audio_path: str | Path, language: str | None=None) -> dict[str, Any]:
        wave,sr=load_audio(audio_path)
        options={"return_timestamps": True}
        if language: options["generate_kwargs"]={"language":language,"task":"transcribe"}
        result=self._load()({"array":wave.numpy(),"sampling_rate":sr},**options)
        return {"text":result["text"].strip(),"chunks":result.get("chunks",[]),"language":language}
