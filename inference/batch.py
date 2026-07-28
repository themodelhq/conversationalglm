from __future__ import annotations
import argparse,asyncio,json
from pathlib import Path
from inference import InferenceEngine, GenerationOptions
async def run(args):
 engine=InferenceEngine(args.model);rows=[json.loads(line) for line in Path(args.input).read_text().splitlines()];output=[]
 for row in rows:output.append({**row,"response":await engine.complete(row["messages"],GenerationOptions(max_new_tokens=args.max_new_tokens))})
 Path(args.output).write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in output)+"\n")
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--input",required=True);p.add_argument("--output",required=True);p.add_argument("--max-new-tokens",type=int,default=256);a=p.parse_args();asyncio.run(run(a))
if __name__=="__main__":main()
