from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import Memory
from memory.embeddings import cosine, embed_text

class LongTermMemory:
    def __init__(self, session: AsyncSession): self.session=session
    async def remember(self,user_id:int,content:str,kind:str="fact",importance:float=.5,metadata:dict | None=None)->Memory:
        if not content.strip(): raise ValueError("memory content must not be empty")
        memory=Memory(id=str(uuid4()),user_id=user_id,content=content.strip(),kind=kind,importance=max(0,min(1,importance)),embedding=embed_text(content),metadata_json=metadata or {})
        self.session.add(memory); await self.session.commit(); await self.session.refresh(memory); return memory
    async def recall(self,user_id:int,query:str,limit:int=6,min_score:float=.18)->list[dict]:
        result=await self.session.execute(select(Memory).where(Memory.user_id==user_id)); vector=embed_text(query); scored=[]
        now=datetime.now(timezone.utc)
        for memory in result.scalars():
            score=.72*cosine(vector,memory.embedding)+.28*memory.importance; 
            if score>=min_score: scored.append((score,memory))
        scored.sort(key=lambda pair:pair[0],reverse=True); chosen=[]
        for score,memory in scored[:limit]:
            memory.last_accessed_at=now; chosen.append({"id":memory.id,"content":memory.content,"kind":memory.kind,"score":round(score,4),"metadata":memory.metadata_json})
        await self.session.commit(); return chosen
    async def forget(self,user_id:int,memory_id:str)->bool:
        memory=await self.session.get(Memory,memory_id)
        if memory is None or memory.user_id!=user_id:return False
        await self.session.delete(memory); await self.session.commit(); return True
