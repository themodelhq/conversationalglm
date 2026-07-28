from __future__ import annotations
import argparse
from pathlib import Path
import torch
from models import ConversationalGLM
class Wrapper(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,input_ids,attention_mask):return self.model(input_ids,attention_mask).logits
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--output",required=True);p.add_argument("--opset",type=int,default=17);a=p.parse_args();model=ConversationalGLM.from_pretrained(a.model).eval();ids=torch.ones((1,8),dtype=torch.long);mask=torch.ones_like(ids);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);torch.onnx.export(Wrapper(model),(ids,mask),out,input_names=["input_ids","attention_mask"],output_names=["logits"],dynamic_axes={"input_ids":{0:"batch",1:"sequence"},"attention_mask":{0:"batch",1:"sequence"},"logits":{0:"batch",1:"sequence"}},opset_version=a.opset,dynamo=False)
if __name__=="__main__":main()
