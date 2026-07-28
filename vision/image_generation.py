from __future__ import annotations
from pathlib import Path
import torch

class ImageGenerator:
    def __init__(self, model_id: str="stabilityai/stable-diffusion-xl-base-1.0", device: str | None=None):
        self.model_id=model_id; self.device=device or ("cuda" if torch.cuda.is_available() else "cpu"); self._pipeline=None
    def _load(self):
        if self._pipeline is None:
            from diffusers import AutoPipelineForText2Image
            dtype=torch.float16 if self.device.startswith("cuda") else torch.float32
            self._pipeline=AutoPipelineForText2Image.from_pretrained(self.model_id,torch_dtype=dtype,use_safetensors=True)
            self._pipeline.enable_attention_slicing(); self._pipeline.to(self.device)
        return self._pipeline
    def generate(self, prompt: str, output: str | Path, negative_prompt: str="", width: int=1024, height: int=1024, steps: int=30, guidance: float=7.0, seed: int | None=None) -> Path:
        if not prompt.strip(): raise ValueError("prompt must not be empty")
        generator=torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        image=self._load()(prompt=prompt,negative_prompt=negative_prompt,width=width,height=height,num_inference_steps=steps,guidance_scale=guidance,generator=generator).images[0]
        output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); image.save(output); return output
