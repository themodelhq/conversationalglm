from __future__ import annotations
import argparse
import numpy as np
import onnxruntime as ort
import torch
from models import ConversationalGLM
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--onnx",required=True);p.add_argument("--atol",type=float,default=2e-3);a=p.parse_args();model=ConversationalGLM.from_pretrained(a.model).eval();ids=torch.randint(0,model.config.vocab_size,(1,16));mask=torch.ones_like(ids);expected=model(ids,mask).logits.detach().numpy();actual=ort.InferenceSession(a.onnx,providers=["CPUExecutionProvider"]).run(["logits"],{"input_ids":ids.numpy(),"attention_mask":mask.numpy()})[0];difference=float(np.max(np.abs(expected-actual)));print({"max_abs_difference":difference,"passed":difference<=a.atol});raise SystemExit(0 if difference<=a.atol else 1)
if __name__=="__main__":main()
