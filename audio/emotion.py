from __future__ import annotations
import torch
from torch import Tensor, nn

EMOTIONS=("neutral","joy","sadness","anger","fear","surprise","disgust","calm")
class SpeechEmotionRecognizer(nn.Module):
    def __init__(self, n_mels: int=80, hidden: int=256, classes: int=len(EMOTIONS)):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(n_mels,hidden,5,padding=2),nn.GELU(),nn.BatchNorm1d(hidden),nn.Conv1d(hidden,hidden,5,padding=2),nn.GELU()); self.attention=nn.Sequential(nn.Conv1d(hidden,1,1),nn.Softmax(dim=-1)); self.classifier=nn.Linear(hidden,classes)
    def forward(self, mel: Tensor) -> Tensor:
        x=self.net(mel.transpose(1,2)); weights=self.attention(x); return self.classifier((x*weights).sum(-1))
    @torch.inference_mode()
    def predict(self, mel: Tensor) -> dict[str,float]:
        probs=torch.softmax(self(mel),-1)[0]; return {e:float(probs[i]) for i,e in enumerate(EMOTIONS)}
