from __future__ import annotations
import httpx
class GLMClient:
    def __init__(self,base_url:str="http://localhost:8000",token:str|None=None,timeout:float=120):self.client=httpx.Client(base_url=base_url,headers={"Authorization":f"Bearer {token}"} if token else {},timeout=timeout)
    def register(self,email:str,password:str)->str:return self.client.post("/v1/auth/register",json={"email":email,"password":password}).raise_for_status().json()["access_token"]
    def chat(self,messages:list[dict],**options)->dict:return self.client.post("/v1/chat/completions",json={"messages":messages,**options}).raise_for_status().json()
    def close(self)->None:self.client.close()
