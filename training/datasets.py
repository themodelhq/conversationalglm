from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Iterator
import yaml

def load_manifest(path:str|Path)->dict:return yaml.safe_load(Path(path).read_text())
def download_manifest(manifest:str|Path,output_dir:str|Path)->list[Path]:
    """Download explicitly licensed Hugging Face datasets from a reviewed manifest."""
    from datasets import load_dataset
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True);written=[]
    for source in load_manifest(manifest).get("datasets",[]):
        if not source.get("allowed_for_training",False):continue
        if "hf_id" not in source:continue
        dataset=load_dataset(source["hf_id"],source.get("subset"),split=source.get("split","train"),trust_remote_code=False)
        destination=output/f"{source['name']}.jsonl"
        with destination.open("w",encoding="utf-8") as stream:
            for row in dataset:stream.write(json.dumps(dict(row),ensure_ascii=False)+"\n")
        written.append(destination)
    return written
def sha256(path:str|Path)->str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()
