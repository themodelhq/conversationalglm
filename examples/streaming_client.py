import asyncio,json,websockets
async def main():
 token="replace-with-jwt";uri="ws://localhost:8000/v1/chat/stream"
 async with websockets.connect(uri,additional_headers={"Authorization":f"Bearer {token}"}) as socket:
  await socket.send(json.dumps({"messages":[{"role":"user","content":"Tell me a short story."}]}))
  async for message in socket:
   event=json.loads(message);print(event.get("content",""),end="",flush=True)
   if event["type"] in {"done","error"}:break
asyncio.run(main())
