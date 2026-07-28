from __future__ import annotations
import argparse
from pathlib import Path
import torch
from models import ConversationalGLM, GLMConfig
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",default="logs/profile");p.add_argument("--steps",type=int,default=10);a=p.parse_args();device="cuda" if torch.cuda.is_available() else "cpu";model=ConversationalGLM(GLMConfig(vocab_size=1024,hidden_size=256,intermediate_size=1024,num_hidden_layers=4,num_attention_heads=4)).to(device);ids=torch.randint(0,1024,(2,256),device=device);optimizer=torch.optim.AdamW(model.parameters())
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]+([torch.profiler.ProfilerActivity.CUDA] if device=="cuda" else []),on_trace_ready=torch.profiler.tensorboard_trace_handler(a.output),record_shapes=True,profile_memory=True) as profiler:
  for _ in range(a.steps):loss=model(ids,labels=ids).loss;loss.backward();optimizer.step();optimizer.zero_grad();profiler.step()
if __name__=="__main__":main()
