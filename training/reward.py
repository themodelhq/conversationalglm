from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from accelerate import Accelerator
from models import ConversationalGLM, ConversationTokenizer
class PreferenceDataset(Dataset):
 def __init__(self,path,tokenizer,max_length):self.rows=[json.loads(x) for x in Path(path).read_text().splitlines()];self.t=tokenizer;self.n=max_length
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];return self.t(r["prompt"]+r["chosen"],truncation=True,max_length=self.n)["input_ids"],self.t(r["prompt"]+r["rejected"],truncation=True,max_length=self.n)["input_ids"]
def collate(items,pad):
 def p(index):return nn.utils.rnn.pad_sequence([torch.tensor(x[index]) for x in items],batch_first=True,padding_value=pad)
 return p(0),p(1)
class RewardModel(nn.Module):
 def __init__(self,base):super().__init__();self.base=base;self.score=nn.Linear(base.config.hidden_size,1)
 def forward(self,ids):
  mask=ids.ne(self.base.config.pad_token_id);hidden=self.base(ids,mask).hidden_states;return self.score(hidden[torch.arange(ids.shape[0],device=ids.device),mask.sum(1)-1]).squeeze(-1)
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--train",required=True);p.add_argument("--output",required=True);p.add_argument("--epochs",type=int,default=1);a=p.parse_args();ac=Accelerator();tokenizer=ConversationTokenizer.load(a.model);model=RewardModel(ConversationalGLM.from_pretrained(a.model));loader=DataLoader(PreferenceDataset(a.train,tokenizer,model.base.config.max_position_embeddings),batch_size=2,shuffle=True,collate_fn=lambda x:collate(x,tokenizer.pad_token_id));optimizer=torch.optim.AdamW(model.parameters(),2e-5);model,loader,optimizer=ac.prepare(model,loader,optimizer)
 for _ in range(a.epochs):
  for chosen,rejected in loader:
   loss=-nn.functional.logsigmoid(model(chosen)-model(rejected)).mean();ac.backward(loss);optimizer.step();optimizer.zero_grad()
 ac.wait_for_everyone();
 if ac.is_main_process:Path(a.output).mkdir(parents=True,exist_ok=True);torch.save(ac.unwrap_model(model).state_dict(),Path(a.output)/"reward_model.pt")
if __name__=="__main__":main()
