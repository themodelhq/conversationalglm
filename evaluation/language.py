from __future__ import annotations
import argparse,json,math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models import ConversationalGLM,ConversationTokenizer
from training.data import JsonlDataset,collate_sft
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--data",required=True);p.add_argument("--output",default="evaluation/language_results.json");a=p.parse_args();model=ConversationalGLM.from_pretrained(a.model);tokenizer=ConversationTokenizer.load(a.model);data=DataLoader(JsonlDataset(a.data,tokenizer,model.config.max_position_embeddings),batch_size=1,collate_fn=lambda x:collate_sft(x,tokenizer.pad_token_id));losses=[]
 with torch.no_grad():
  for batch in data:losses.append(float(model(**batch).loss))
 result={"cross_entropy":sum(losses)/len(losses),"perplexity":math.exp(min(20,sum(losses)/len(losses))),"examples":len(losses)};Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=="__main__":main()
