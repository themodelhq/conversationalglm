from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator
import torch
from models import ConversationalGLM, ConversationTokenizer, render_messages
from models.configuration_glm import GLMConfig
from inference.functions import FunctionRegistry, parse_tool_call

@dataclass
class GenerationOptions:
    max_new_tokens:int=512
    temperature:float=.7
    top_p:float=.9
    system_prompt:str="You are a helpful, accurate, safe multimodal assistant. Respond in the user's language."
logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self,model_path:str|None=None,device:str="auto",max_concurrent:int=4):
        self.device=self._device(device); self.model:ConversationalGLM|None=None; self.tokenizer=None; self.model_error:str|None=None; self.semaphore=asyncio.Semaphore(max_concurrent); self.functions=FunctionRegistry()
        if model_path:
            try:self.load(model_path)
            except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
                self.model_error=str(error); logger.warning("Model was not loaded; the API will start in platform-only mode: %s",error)
    @staticmethod
    def _device(device:str)->str:return "cuda" if device=="auto" and torch.cuda.is_available() else ("cpu" if device=="auto" else device)
    @property
    def ready(self)->bool:return self.model is not None and self.tokenizer is not None
    def load(self,path:str|Path)->None:
        path=Path(path).expanduser()
        required=("config.json","model.safetensors","tokenizer.json")
        if not path.is_dir():raise FileNotFoundError(f"Configured model directory does not exist: {path}")
        missing=[name for name in required if not (path/name).is_file()]
        if missing:raise FileNotFoundError(f"Configured model directory is incomplete; missing: {', '.join(missing)}")
        tokenizer=ConversationTokenizer.load(path); model=ConversationalGLM.from_pretrained(path,self.device); model.eval()
        self.tokenizer=tokenizer; self.model=model; self.model_error=None
    def _generate_sync(self,messages:list[dict[str,str]],options:GenerationOptions)->str:
        if not self.ready:raise RuntimeError("No trained model is loaded. Set GLM_MODEL_PATH to an exported model directory.")
        prompt=render_messages(messages,True); encoded=self.tokenizer(prompt,return_tensors="pt",truncation=True,max_length=self.model.config.max_position_embeddings).to(self.device)
        output=self.model.generate(encoded.input_ids,encoded.attention_mask,options.max_new_tokens,options.temperature,options.top_p); generated=self.tokenizer.decode(output[0,encoded.input_ids.shape[1]:],skip_special_tokens=False)
        return generated.replace("<eos>","").strip()
    async def complete(self,messages:list[dict[str,str]],options:GenerationOptions|None=None)->dict:
        options=options or GenerationOptions(); messages=[{"role":"system","content":options.system_prompt},*messages] if not any(m.get("role")=="system" for m in messages) else messages
        async with self.semaphore:
            text=await asyncio.to_thread(self._generate_sync,messages,options); tool=parse_tool_call(text)
            calls=[]
            if tool:
                name,args=tool; result=self.functions.call(name,args); calls.append({"name":name,"arguments":args,"result":result}); followup=[*messages,{"role":"assistant","content":text},{"role":"tool","content":str(result)}]; text=await asyncio.to_thread(self._generate_sync,followup,options)
        return {"content":text,"tool_calls":calls}
    async def stream(self,messages:list[dict[str,str]],options:GenerationOptions|None=None)->AsyncIterator[str]:
        result=await self.complete(messages,options)
        for token in result["content"].split(" "):
            yield token+" "
