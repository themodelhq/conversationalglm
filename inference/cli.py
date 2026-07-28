from __future__ import annotations
import argparse,asyncio
from inference import GenerationOptions, InferenceEngine
async def run():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--prompt",required=True);p.add_argument("--max-new-tokens",type=int,default=256);a=p.parse_args();engine=InferenceEngine(a.model);result=await engine.complete([{"role":"user","content":a.prompt}],GenerationOptions(max_new_tokens=a.max_new_tokens));print(result["content"])
if __name__=="__main__":asyncio.run(run())
