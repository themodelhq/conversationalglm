from __future__ import annotations
import hashlib
import re
import numpy as np

def embed_text(text: str, dimensions: int=384) -> list[float]:
    """Stable hashed n-gram embedding; no network model is required for persistent retrieval."""
    vector=np.zeros(dimensions,dtype=np.float32); normalized=" "+re.sub(r"\s+"," ",text.lower()).strip()+" "
    for n in (2,3,4,5):
        for i in range(max(0,len(normalized)-n+1)):
            digest=hashlib.blake2b(normalized[i:i+n].encode(),digest_size=8).digest(); value=int.from_bytes(digest,"big"); vector[value%dimensions]+=1 if value&1 else -1
    norm=np.linalg.norm(vector); return (vector/(norm or 1)).tolist()
def cosine(a: list[float], b: list[float]) -> float:
    x,y=np.asarray(a,dtype=np.float32),np.asarray(b,dtype=np.float32); return float(x@y/(np.linalg.norm(x)*np.linalg.norm(y)+1e-8))
