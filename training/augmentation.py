from __future__ import annotations
import random
import torch
from torch import Tensor

def augment_text(text:str,drop_probability:float=0.03,seed:int|None=None)->str:
    """Low-rate token dropout for robust denoising objectives; preserves at least one token."""
    words=text.split()
    if len(words)<3 or drop_probability<=0:return text
    rng=random.Random(seed);kept=[word for word in words if rng.random()>=drop_probability]
    return " ".join(kept or words[:1])
def augment_waveform(waveform:Tensor,gain_db:float|None=None,noise_scale:float=0.002)->Tensor:
    gain_db=random.uniform(-4,4) if gain_db is None else gain_db;gain=10**(gain_db/20);noise=torch.randn_like(waveform)*noise_scale
    return (waveform*gain+noise).clamp(-1,1)
def augment_mel(mel:Tensor,time_masks:int=2,frequency_masks:int=2,max_time_width:int=24,max_frequency_width:int=8)->Tensor:
    output=mel.clone();time,frequency=output.shape[-2:]
    for _ in range(time_masks):
        width=random.randint(0,min(max_time_width,time));start=random.randint(0,max(0,time-width));output[...,start:start+width,:]=0
    for _ in range(frequency_masks):
        width=random.randint(0,min(max_frequency_width,frequency));start=random.randint(0,max(0,frequency-width));output[..., :,start:start+width]=0
    return output
def augment_image(pixels:Tensor,flip_probability:float=.5,color_jitter:float=.08)->Tensor:
    output=pixels.clone()
    if random.random()<flip_probability:output=output.flip(-1)
    return output*(1+random.uniform(-color_jitter,color_jitter))
