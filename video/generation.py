from __future__ import annotations
from pathlib import Path
import torch

class VideoGenerator:
    def __init__(self, model_id: str="damo-vilab/text-to-video-ms-1.7b", device: str | None=None):
        self.model_id=model_id; self.device=device or ("cuda" if torch.cuda.is_available() else "cpu"); self._pipeline=None
    def _load(self):
        if self._pipeline is None:
            from diffusers import DiffusionPipeline
            dtype=torch.float16 if self.device.startswith("cuda") else torch.float32
            self._pipeline=DiffusionPipeline.from_pretrained(self.model_id,torch_dtype=dtype); self._pipeline.enable_vae_slicing(); self._pipeline.to(self.device)
        return self._pipeline
    def generate(self, prompt: str, output: str | Path, frames: int=24, fps: int=8, seed: int | None=None) -> Path:
        if not prompt.strip(): raise ValueError("prompt must not be empty")
        gen=torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        result=self._load()(prompt,num_frames=frames,generator=gen)
        from diffusers.utils import export_to_video
        output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); export_to_video(result.frames[0],str(output),fps=fps); return output
