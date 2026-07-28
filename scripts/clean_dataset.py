from __future__ import annotations
import argparse,json,re
from pathlib import Path
def clean(text:str)->str:return re.sub(r"\s+"," ",re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]","",text)).strip()
def transform(value):
 if isinstance(value,str):return clean(value)
 if isinstance(value,list):return [transform(x) for x in value]
 if isinstance(value,dict):return {k:transform(v) for k,v in value.items()}
 return value
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True);a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 with Path(a.input).open(encoding="utf-8") as source,out.open("w",encoding="utf-8") as destination:
  for line in source:
   try:row=transform(json.loads(line))
   except json.JSONDecodeError:continue
   if row:destination.write(json.dumps(row,ensure_ascii=False)+"\n")
if __name__=="__main__":main()
