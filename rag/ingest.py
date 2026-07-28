from __future__ import annotations
from pathlib import Path
from rag.retriever import Retriever
async def ingest_file(retriever:Retriever,path:str|Path,user_id:int|None=None)->list[str]:
    path=Path(path); suffix=path.suffix.lower()
    if suffix in {".txt",".md",".csv",".json",".jsonl"}: text=path.read_text(encoding="utf-8",errors="replace")
    elif suffix==".pdf":
        from pypdf import PdfReader
        text="\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    else: raise ValueError(f"Unsupported document extension: {suffix}")
    return await retriever.index(str(path),path.stem,text,user_id)
