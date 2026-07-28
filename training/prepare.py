from __future__ import annotations
import argparse,json,random,re
from collections import Counter,defaultdict
from pathlib import Path
import yaml
from models.tokenizer import ConversationTokenizer

def valid(row:dict,min_chars:int)->bool:
    text=json.dumps(row.get("messages",row.get("text",row.get("prompt",""))),ensure_ascii=False) + str(row.get("response",""))
    return len(text.strip())>=min_chars and not re.search(r"\x00",text)
def normalize(value):
    if isinstance(value,str):return re.sub(r"\s+"," ",value).strip()
    if isinstance(value,list):return [{**m,"content":normalize(str(m.get("content","")))} if isinstance(m,dict) else m for m in value]
    return value
def augment(row:dict)->dict:
    result=dict(row)
    if "messages" in result:result["messages"]=normalize(result["messages"])
    for key in ("text","prompt","response","chosen","rejected"):
        if key in result:result[key]=normalize(str(result[key]))
    return result
def records(paths:list[Path],minimum:int):
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:row=augment(json.loads(line))
                except json.JSONDecodeError:continue
                if valid(row,minimum):yield row
def row_text(row:dict)->str:
    if "messages" in row:return " ".join(str(m.get("content","")) for m in row["messages"] if isinstance(m,dict))
    return " ".join(str(row.get(k,"")) for k in ("text","prompt","response","chosen","rejected"))
def balance_rows(rows:list[dict],key:str,seed:int)->list[dict]:
    groups:dict[str,list[dict]]=defaultdict(list)
    for row in rows:groups[str(row.get(key,"unknown"))].append(row)
    if len(groups)<2:return rows
    rng=random.Random(seed); target=max(len(items) for items in groups.values());result=[]
    for items in groups.values():
        result.extend(items);result.extend(rng.choice(items) for _ in range(target-len(items)))
    rng.shuffle(result);return result
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="config/data.yaml");parser.add_argument("--input",nargs="*");args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())["data"];output=Path(cfg["output_dir"]);output.mkdir(parents=True,exist_ok=True)
    paths=[Path(item) for item in (args.input or cfg.get("sources",[]))]
    if not paths:raise SystemExit("Provide --input JSONL files or data.sources in config")
    rows=list(records(paths,cfg.get("min_text_chars",2)));random.Random(cfg.get("seed",42)).shuffle(rows)
    if not rows:raise SystemExit("No valid records found")
    ratio=float(cfg.get("validation_ratio",.02));cut=max(1,round(len(rows)*(1-ratio)));train,validation=rows[:cut],rows[cut:]
    if not validation:validation=train[-1:];train=train[:-1]
    if cfg.get("balance_by"):train=balance_rows(train,str(cfg["balance_by"]),int(cfg.get("seed",42)))
    for name,items in (("train",train),("validation",validation)): 
        with (output/f"{name}.jsonl").open("w",encoding="utf-8") as stream:
            for row in items:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    shard_size=int(cfg.get("shard_size",10000));shards=output/"shards";shards.mkdir(exist_ok=True)
    for start in range(0,len(train),shard_size):
        with (shards/f"train-{start//shard_size:05d}.jsonl").open("w",encoding="utf-8") as stream:
            for row in train[start:start+shard_size]:stream.write(json.dumps(row,ensure_ascii=False)+"\n")
    ConversationTokenizer.train((row_text(row) for row in train),cfg["tokenizer_dir"])
    (output/"stats.json").write_text(json.dumps({"train":len(train),"validation":len(validation),"shards":(len(train)+shard_size-1)//shard_size,"languages":dict(Counter(str(r.get("language","unknown")) for r in rows))},indent=2))
if __name__=="__main__":main()
