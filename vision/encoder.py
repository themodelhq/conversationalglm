from __future__ import annotations
from pathlib import Path
import torch
from torch import Tensor, nn
from PIL import Image
import numpy as np

class VisionEncoder(nn.Module):
    """ViT-compatible patch encoder that emits language-projectable image tokens."""
    def __init__(self, image_size: int = 224, patch_size: int = 16, hidden_size: int = 768, layers: int = 12, heads: int = 12):
        super().__init__(); self.image_size=image_size; self.patch_size=patch_size
        patches=(image_size//patch_size)**2; self.patch=nn.Conv2d(3,hidden_size,patch_size,patch_size); self.cls=nn.Parameter(torch.zeros(1,1,hidden_size)); self.pos=nn.Parameter(torch.zeros(1,patches+1,hidden_size))
        block=nn.TransformerEncoderLayer(hidden_size,heads,hidden_size*4,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(block,layers); self.norm=nn.LayerNorm(hidden_size); nn.init.trunc_normal_(self.pos,std=.02); nn.init.trunc_normal_(self.cls,std=.02)
    def forward(self, pixels: Tensor) -> Tensor:
        x=self.patch(pixels).flatten(2).transpose(1,2); x=torch.cat((self.cls.expand(x.shape[0],-1,-1),x),1); return self.norm(self.encoder(x+self.pos))
    @staticmethod
    def preprocess(image: Image.Image, size: int=224) -> Tensor:
        image=image.convert("RGB").resize((size,size)); array=np.asarray(image,dtype=np.float32)/255.0; array=(array-np.array([.485,.456,.406]))/np.array([.229,.224,.225]); return torch.from_numpy(array.transpose(2,0,1)).float()
    @torch.inference_mode()
    def encode_file(self, path: str | Path, device: str | torch.device="cpu") -> Tensor:
        return self(self.preprocess(Image.open(path),self.image_size).unsqueeze(0).to(device))
