from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa
import torch
from torch import Tensor

def load_audio(path: str | Path, sample_rate: int=16000) -> tuple[Tensor,int]:
    wave,sr=librosa.load(str(path),sr=sample_rate,mono=True); return torch.tensor(wave,dtype=torch.float32),sr

def log_mel(waveform: Tensor, sample_rate: int=16000, n_mels: int=80, hop_length: int=160) -> Tensor:
    audio=waveform.detach().cpu().numpy(); mel=librosa.feature.melspectrogram(y=audio,sr=sample_rate,n_fft=400,hop_length=hop_length,n_mels=n_mels,power=2.0); return torch.tensor(librosa.power_to_db(mel,ref=np.max),dtype=torch.float32).transpose(0,1)
