from __future__ import annotations
import argparse
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--lines-per-shard",type=int,default=10000);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);handle=None
 try:
  for index,line in enumerate(Path(a.input).open(encoding="utf-8")):
   if index%a.lines_per_shard==0:
    if handle:handle.close()
    handle=(out/f"shard-{index//a.lines_per_shard:05d}.jsonl").open("w",encoding="utf-8")
   handle.write(line)
 finally:
  if handle:handle.close()
if __name__=="__main__":main()
