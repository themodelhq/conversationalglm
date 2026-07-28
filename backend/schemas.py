from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator
class RegisterRequest(BaseModel): email:EmailStr; password:str=Field(min_length=12,max_length=128)
class LoginRequest(RegisterRequest): pass
class TokenResponse(BaseModel): access_token:str; token_type:str="bearer"
class UserResponse(BaseModel): model_config=ConfigDict(from_attributes=True); id:int; email:EmailStr; is_active:bool
class MessageInput(BaseModel): role:str=Field(pattern="^(system|user|assistant|tool)$"); content:str=Field(min_length=1,max_length=100000)
class ChatRequest(BaseModel): messages:list[MessageInput]=Field(min_length=1,max_length=128); conversation_id:str|None=None; max_new_tokens:int=Field(default=512,ge=1,le=4096); temperature:float=Field(default=.7,ge=0,le=2); top_p:float=Field(default=.9,gt=0,le=1); use_memory:bool=True; use_rag:bool=True
class ChatResponse(BaseModel): conversation_id:str; content:str; memories:list[dict]=[]; sources:list[dict]=[]; tool_calls:list[dict]=[]
class MemoryRequest(BaseModel): content:str=Field(min_length=1,max_length=5000); kind:str="fact"; importance:float=Field(default=.5,ge=0,le=1)
class DocumentRequest(BaseModel): title:str=Field(min_length=1,max_length=512); source:str=Field(default="api",max_length=1024); content:str=Field(min_length=1,max_length=2_000_000)
class ASRResponse(BaseModel): text:str; chunks:list[dict]=[]; language:str|None=None
class TTSRequest(BaseModel): text:str=Field(min_length=1,max_length=5000); language:str="en"; emotion:str="neutral"; speaker_wav:str|None=None
class ImageGenerationRequest(BaseModel): prompt:str=Field(min_length=1,max_length=4000); negative_prompt:str=""; width:int=Field(default=1024,ge=256,le=2048,multiple_of=8); height:int=Field(default=1024,ge=256,le=2048,multiple_of=8); steps:int=Field(default=30,ge=1,le=100); guidance:float=Field(default=7.0,ge=0,le=30); seed:int|None=None
class VideoGenerationRequest(BaseModel): prompt:str=Field(min_length=1,max_length=4000); frames:int=Field(default=24,ge=8,le=120); fps:int=Field(default=8,ge=1,le=30); seed:int|None=None
class AvatarRequest(BaseModel): visual_prompt:str=Field(min_length=1,max_length=4000); speech_audio_path:str=Field(min_length=1,max_length=1024); spoken_text:str=Field(min_length=1,max_length=5000); emotion:str="neutral"; duration:float=Field(default=4,gt=0,le=30)
class DatasetResponse(BaseModel):
    model_config=ConfigDict(from_attributes=True)
    id:str; name:str; format:str; task:str; records:int; size_bytes:int; created_at:datetime
class TrainingLaunchRequest(BaseModel):
    name:str=Field(min_length=3,max_length=120)
    dataset_id:str
    task:str=Field(default="sft",pattern="^(sft|dpo|asr|tts|emotion_recognition|emotion_generation|vision|video|motion|lipsync|gesture|memory)$")
    model_size:str=Field(default="starter",pattern="^(starter|standard|advanced)$")
    epochs:int=Field(default=3,ge=1,le=100)
    batch_size:int=Field(default=1,ge=1,le=128)
    gradient_accumulation_steps:int=Field(default=16,ge=1,le=256)
    learning_rate:float=Field(default=0.0002,gt=0,le=0.01)
    max_length:int=Field(default=2048,ge=128,le=8192)
    mixed_precision:str=Field(default="bf16",pattern="^(no|fp16|bf16)$")
    strategy:str=Field(default="single",pattern="^(single|deepspeed|fsdp)$")
    gpu_count:int=Field(default=1,ge=1,le=64)
    node_count:int=Field(default=1,ge=1,le=64)
    machine_rank:int=Field(default=0,ge=0,le=63)
    main_process_ip:str=Field(default="127.0.0.1",min_length=1,max_length=253)
    main_process_port:int=Field(default=29500,ge=1024,le=65535)
    tracking:str=Field(default="tensorboard",pattern="^(none|tensorboard|wandb)$")
    warmup_ratio:float=Field(default=.03,ge=0,le=.5)
    weight_decay:float=Field(default=.1,ge=0,le=1)
    save_steps:int=Field(default=500,ge=10,le=100000)
    eval_steps:int=Field(default=250,ge=10,le=100000)
    @model_validator(mode="after")
    def validate_distribution(self):
        if self.machine_rank >= self.node_count: raise ValueError("machine_rank must be smaller than node_count")
        if self.strategy == "single" and self.node_count > 1: raise ValueError("multi-node runs require DeepSpeed or FSDP")
        return self
class TrainingRunResponse(BaseModel):
    model_config=ConfigDict(from_attributes=True)
    id:str; name:str; task:str; status:str; dataset_id:str; config_json:dict; metrics_json:dict; output_dir:str; process_id:int|None; return_code:int|None; created_at:datetime; started_at:datetime|None; finished_at:datetime|None
class TrainingLogResponse(BaseModel): id:str; status:str; lines:list[str]
