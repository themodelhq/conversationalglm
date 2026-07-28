from __future__ import annotations
from pathlib import Path
from uuid import uuid4
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import Document
from memory.embeddings import cosine, embed_text

def chunk_text(text:str,chunk_size:int=900,overlap:int=150)->list[str]:
    words=text.split(); chunks=[]
    for start in range(0,len(words),max(1,chunk_size-overlap)):
        piece=" ".join(words[start:start+chunk_size])
        if piece: chunks.append(piece)
        if start+chunk_size>=len(words):break
    return chunks
class Retriever:
    def __init__(self,session:AsyncSession):self.session=session
    async def index(self,source:str,title:str,text:str,user_id:int | None=None,metadata:dict | None=None)->list[str]:
        if not text.strip(): raise ValueError("document text must not be empty")
        ids=[]
        for number,part in enumerate(chunk_text(text)):
            document=Document(id=str(uuid4()),user_id=user_id,source=source,title=title,content=part,embedding=embed_text(part),metadata_json={**(metadata or {}),"chunk":number})
            self.session.add(document); ids.append(document.id)
        await self.session.commit(); return ids
    async def search(self,query:str,user_id:int | None=None,limit:int=5)->list[dict]:
        condition=or_(Document.user_id.is_(None),Document.user_id==user_id) if user_id is not None else Document.user_id.is_(None)
        result=await self.session.execute(select(Document).where(condition)); q=embed_text(query); ranked=sorted(((cosine(q,d.embedding),d) for d in result.scalars()),key=lambda pair:pair[0],reverse=True)
        return [{"id":d.id,"title":d.title,"source":d.source,"content":d.content,"score":round(score,4),"metadata":d.metadata_json} for score,d in ranked[:limit] if score>.1]
