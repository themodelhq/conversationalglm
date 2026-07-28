from __future__ import annotations
from pathlib import Path
import torch
from PIL import Image
class ImageUnderstanding:
    """Open image-to-text captioning backend for image understanding requests."""
    def __init__(self,model_id:str="Salesforce/blip-image-captioning-base",device:str|None=None):
        self.model_id=model_id;self.device=device or ("cuda:0" if torch.cuda.is_available() else "cpu");self._pipeline=None
    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline=pipeline("image-to-text",model=self.model_id,device=0 if self.device.startswith("cuda") else -1)
        return self._pipeline
    def describe(self,image_path:str|Path,prompt:str|None=None)->str:
        image=Image.open(image_path).convert("RGB");kwargs={"max_new_tokens":128}
        if prompt:kwargs["prompt"]=prompt
        result=self._load()(image,**kwargs)
        return str(result[0].get("generated_text","")).strip()
