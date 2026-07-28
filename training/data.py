from __future__ import annotations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from models.tokenizer import render_messages

@dataclass
class Example:
    input_ids:list[int]
    labels:list[int]
    attention_mask:list[int]
class JsonlDataset(Dataset):
    def __init__(self,path:str|Path,tokenizer,max_length:int,task:str="sft"):
        self.path=Path(path);self.tokenizer=tokenizer;self.max_length=max_length;self.task=task
        if not self.path.exists():raise FileNotFoundError(f"Dataset not found: {self.path}")
        self.offsets=[]
        with self.path.open("rb") as stream:
            while stream.readline():self.offsets.append(stream.tell())
        if not self.offsets:raise ValueError(f"Dataset is empty: {self.path}")
    def __len__(self):return len(self.offsets)
    def _line(self,index:int)->dict:
        start=0 if index==0 else self.offsets[index-1]
        with self.path.open(encoding="utf-8") as stream:stream.seek(start);return json.loads(stream.readline())
    def _text(self,row:dict)->tuple[str,str]:
        if "messages" in row:
            messages=row["messages"]; prompt=render_messages(messages[:-1],True) if messages and messages[-1].get("role")=="assistant" else render_messages(messages,True); target=messages[-1].get("content","") if messages and messages[-1].get("role")=="assistant" else row.get("response","");return prompt,target
        return str(row.get("prompt",row.get("text",""))),str(row.get("response",row.get("text","")))
    def __getitem__(self,index:int)->dict[str,Any]:
        row=self._line(index)
        if self.task=="dpo":
            prompt=str(row["prompt"]); chosen=str(row["chosen"]); rejected=str(row["rejected"])
            return {"chosen":self.tokenizer(prompt+chosen,truncation=True,max_length=self.max_length)["input_ids"],"rejected":self.tokenizer(prompt+rejected,truncation=True,max_length=self.max_length)["input_ids"]}
        prompt,target=self._text(row); full=prompt+target+self.tokenizer.eos_token; encoded=self.tokenizer(full,truncation=True,max_length=self.max_length)
        prompt_len=len(self.tokenizer(prompt,truncation=True,max_length=self.max_length)["input_ids"]);labels=encoded["input_ids"].copy();labels[:prompt_len]=[-100]*min(prompt_len,len(labels))
        return {"input_ids":encoded["input_ids"],"attention_mask":encoded["attention_mask"],"labels":labels}
class DistributedShardSampler(Sampler[int]):
    """Deterministic rank partitioning for pre-sharded JSONL datasets."""
    def __init__(self,dataset:Dataset,rank:int=0,world_size:int=1,shuffle:bool=True,seed:int=42):
        self.dataset=dataset;self.rank=rank;self.world_size=world_size;self.shuffle=shuffle;self.seed=seed;self.epoch=0
    def set_epoch(self,epoch:int)->None:self.epoch=epoch
    def __iter__(self):
        indices=list(range(len(self.dataset)))
        if self.shuffle:
            generator=torch.Generator();generator.manual_seed(self.seed+self.epoch);indices=torch.randperm(len(indices),generator=generator).tolist()
        return iter(indices[self.rank::self.world_size])
    def __len__(self)->int:return max(0,(len(self.dataset)-self.rank+self.world_size-1)//self.world_size)
def collate_sft(batch:list[dict],pad_token_id:int)->dict[str,torch.Tensor]:
    def pad(key:str,value:int):return pad_sequence([torch.tensor(x[key],dtype=torch.long) for x in batch],batch_first=True,padding_value=value)
    return {"input_ids":pad("input_ids",pad_token_id),"attention_mask":pad("attention_mask",0),"labels":pad("labels",-100)}
def collate_dpo(batch:list[dict],pad_token_id:int)->dict[str,torch.Tensor]:
    def pad(key:str):return pad_sequence([torch.tensor(row[key],dtype=torch.long) for row in batch],batch_first=True,padding_value=pad_token_id)
    return {"chosen":pad("chosen"),"rejected":pad("rejected")}
