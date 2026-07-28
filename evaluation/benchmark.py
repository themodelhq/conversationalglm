from __future__ import annotations
import argparse,asyncio,json,time
from pathlib import Path
import psutil
from inference import InferenceEngine,GenerationOptions
async def run(args):
 engine=InferenceEngine(args.model);durations=[]
 for _ in range(args.runs):
  start=time.perf_counter();await engine.complete([{ "role":"user", "content":args.prompt}],GenerationOptions(max_new_tokens=args.max_new_tokens));durations.append(time.perf_counter()-start)
 result={"runs":args.runs,"mean_seconds":sum(durations)/len(durations),"p95_seconds":sorted(durations)[max(0,int(len(durations)*.95)-1)],"rss_mb":psutil.Process().memory_info().rss/1024**2};Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--output",default="evaluation/benchmark.json");p.add_argument("--runs",type=int,default=10);p.add_argument("--prompt",default="Explain retrieval augmented generation in one sentence.");p.add_argument("--max-new-tokens",type=int,default=64);a=p.parse_args();asyncio.run(run(a))
if __name__=="__main__":main()
