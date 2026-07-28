from __future__ import annotations
import argparse,asyncio,json
from pathlib import Path
from inference import InferenceEngine,GenerationOptions
async def run(args):
 cases=[json.loads(line) for line in Path(args.cases).read_text().splitlines()];engine=InferenceEngine(args.model);results=[]
 for case in cases:
  response=(await engine.complete([{"role":"user","content":case["prompt"]}],GenerationOptions(max_new_tokens=256)))["content"];blocked=any(term.lower() in response.lower() for term in case.get("forbidden",[]));results.append({"id":case.get("id"),"blocked":blocked,"response":response})
 summary={"cases":len(results),"passed":sum(not row["blocked"] for row in results),"results":results};Path(args.output).write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({"cases":summary["cases"],"passed":summary["passed"]}))
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--cases",required=True);p.add_argument("--output",default="evaluation/safety.json");a=p.parse_args();asyncio.run(run(a))
if __name__=="__main__":main()
