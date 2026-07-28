from __future__ import annotations
import asyncio
import json
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from uuid import uuid4
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from audio import SpeechRecognizer, SpeechSynthesizer
from vision import ImageGenerator, ImageUnderstanding
from video import VideoGenerator
from backend.auth import create_token, current_user, hash_password, verify_password
from backend.schemas import ASRResponse, ChatRequest, ChatResponse, DatasetResponse, DocumentRequest, ImageGenerationRequest, LoginRequest, MemoryRequest, RegisterRequest, TTSRequest, TokenResponse, TrainingLaunchRequest, TrainingLogResponse, TrainingRunResponse, UserResponse, VideoGenerationRequest
from backend.settings import settings
from backend.training_service import training_manager
from database.models import Conversation, DatasetAsset, Message, TrainingRun, User
from database.session import get_session, initialize_database
from emotion import detect_text_emotion
from inference import GenerationOptions, InferenceEngine
from memory import LongTermMemory, extract_memories
from rag import Retriever

REQUESTS=Counter("glm_api_requests_total","API requests",["method","path","status"]); LATENCY=Histogram("glm_api_request_duration_seconds","API request duration",["path"]); MODEL_READY=Gauge("glm_model_ready","Model loaded state")
class RateLimiter:
    def __init__(self,limit:int):self.limit=limit;self.events:dict[str,deque[float]]=defaultdict(deque);self.lock=asyncio.Lock()
    async def allow(self,key:str)->bool:
        now=monotonic()
        async with self.lock:
            queue=self.events[key]
            while queue and queue[0]<=now-60:queue.popleft()
            if len(queue)>=self.limit:return False
            queue.append(now);return True
limiter=RateLimiter(settings.api_rate_limit)
@asynccontextmanager
async def lifespan(app:FastAPI):
    state_root=training_manager.initialize_storage(); app.state.storage_root=state_root
    await initialize_database(); await training_manager.recover(); app.state.engine=InferenceEngine(settings.model_path or None,settings.device); MODEL_READY.set(1 if app.state.engine.ready else 0); yield
app=FastAPI(title="Conversational GLM API",version="0.1.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=settings.origins,allow_origin_regex=settings.cors_origin_regex or None,allow_credentials=True,allow_methods=["GET","POST","DELETE"],allow_headers=["Authorization","Content-Type"])
@app.middleware("http")
async def observe(request:Request,call_next):
    started=monotonic(); key=request.client.host if request.client else "unknown"
    if request.url.path not in {"/health","/metrics"} and not await limiter.allow(key):return JSONResponse({"detail":"Rate limit exceeded"},status_code=429)
    response=await call_next(request); path=request.url.path; REQUESTS.labels(request.method,path,str(response.status_code)).inc(); LATENCY.labels(path).observe(monotonic()-started); return response
@app.get("/health")
async def health(request:Request):return {"status":"ok","model_ready":request.app.state.engine.ready,"model_status":"loaded" if request.app.state.engine.ready else ("unavailable" if request.app.state.engine.model_error else "not_configured"),"environment":settings.env,"storage_dir":str(training_manager.state_root),"persistent_storage":not training_manager.using_ephemeral_storage}
@app.get("/metrics")
async def metrics():return StreamingResponse(iter([generate_latest()]),media_type=CONTENT_TYPE_LATEST)
@app.post("/v1/auth/register",response_model=TokenResponse,status_code=status.HTTP_201_CREATED)
async def register(body:RegisterRequest,session:AsyncSession=Depends(get_session)):
    email=str(body.email).lower()
    existing=await session.scalar(select(User).where(User.email==email))
    if existing:raise HTTPException(status.HTTP_409_CONFLICT,"Email is already registered")
    user=User(email=email,password_hash=hash_password(body.password)); session.add(user)
    try:
        await session.commit();await session.refresh(user)
    except IntegrityError:
        await session.rollback();raise HTTPException(status.HTTP_409_CONFLICT,"Email is already registered")
    return TokenResponse(access_token=create_token(user))
@app.post("/v1/auth/login",response_model=TokenResponse)
async def login(body:LoginRequest,session:AsyncSession=Depends(get_session)):
    user=await session.scalar(select(User).where(User.email==str(body.email).lower()))
    if user is None or not verify_password(body.password,user.password_hash):raise HTTPException(status.HTTP_401_UNAUTHORIZED,"Incorrect email or password")
    return TokenResponse(access_token=create_token(user))
@app.get("/v1/me",response_model=UserResponse)
async def me(user:User=Depends(current_user)):return user
async def conversation_messages(session:AsyncSession,conversation_id:str,user_id:int)->list[dict[str,str]]:
    conversation=await session.get(Conversation,conversation_id)
    if conversation is None or conversation.user_id!=user_id:raise HTTPException(status.HTTP_404_NOT_FOUND,"Conversation not found")
    rows=await session.scalars(select(Message).where(Message.conversation_id==conversation_id).order_by(Message.created_at)); return [{"role":m.role,"content":m.content} for m in rows]
@app.post("/v1/chat/completions",response_model=ChatResponse)
async def chat(body:ChatRequest,request:Request,user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    engine:InferenceEngine=request.app.state.engine
    if not engine.ready:raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,"Model is not loaded")
    conversation_id=body.conversation_id or str(uuid4()); conversation=await session.get(Conversation,conversation_id)
    if conversation is None:
        conversation=Conversation(id=conversation_id,user_id=user.id,title=body.messages[0].content[:120]);session.add(conversation);await session.flush()
    elif conversation.user_id!=user.id:raise HTTPException(status.HTTP_404_NOT_FOUND,"Conversation not found")
    history=await conversation_messages(session,conversation_id,user.id) if body.conversation_id else []
    incoming=[message.model_dump() for message in body.messages]; user_text="\n".join(item["content"] for item in incoming if item["role"]=="user")
    memories=await LongTermMemory(session).recall(user.id,user_text) if body.use_memory and user_text else []
    sources=await Retriever(session).search(user_text,user.id) if body.use_rag and user_text else []
    context=[]
    if memories:context.append({"role":"system","content":"Relevant long-term memory:\n"+"\n".join("- "+m["content"] for m in memories)})
    if sources:context.append({"role":"system","content":"Retrieved reference material:\n"+"\n".join(f"[{s['title']}] {s['content']}" for s in sources)})
    result=await engine.complete([*context,*history,*incoming],GenerationOptions(body.max_new_tokens,body.temperature,body.top_p))
    for message in incoming:session.add(Message(id=str(uuid4()),conversation_id=conversation_id,role=message["role"],content=message["content"],metadata_json={}))
    session.add(Message(id=str(uuid4()),conversation_id=conversation_id,role="assistant",content=result["content"],metadata_json={"tool_calls":result["tool_calls"],"emotion":detect_text_emotion(result["content"])}))
    for fact,importance in extract_memories(user_text):await LongTermMemory(session).remember(user.id,fact,importance=importance)
    await session.commit(); return ChatResponse(conversation_id=conversation_id,content=result["content"],memories=memories,sources=sources,tool_calls=result["tool_calls"])
@app.websocket("/v1/chat/stream")
async def chat_stream(websocket:WebSocket):
    await websocket.accept()
    try:
        payload=await websocket.receive_json(); token=websocket.headers.get("authorization","").removeprefix("Bearer ")
        from jose import jwt
        claims=jwt.decode(token,settings.jwt_secret,algorithms=[settings.jwt_algorithm]);
        if not claims.get("sub"):raise ValueError("missing subject")
        messages=payload.get("messages",[]); options=GenerationOptions(max_new_tokens=min(int(payload.get("max_new_tokens",512)),4096),temperature=float(payload.get("temperature",.7)),top_p=float(payload.get("top_p",.9)))
        async for token_text in websocket.app.state.engine.stream(messages,options):await websocket.send_json({"type":"token","content":token_text})
        await websocket.send_json({"type":"done"})
    except (ValueError,KeyError,TypeError) as error:await websocket.send_json({"type":"error","detail":str(error)})
    except WebSocketDisconnect: return
    finally:await websocket.close()
@app.post("/v1/memories")
async def remember(body:MemoryRequest,user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    item=await LongTermMemory(session).remember(user.id,body.content,body.kind,body.importance);return {"id":item.id,"content":item.content}
@app.get("/v1/memories")
async def memories(query:str,user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):return await LongTermMemory(session).recall(user.id,query)
@app.delete("/v1/memories/{memory_id}",status_code=status.HTTP_204_NO_CONTENT)
async def forget(memory_id:str,user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    if not await LongTermMemory(session).forget(user.id,memory_id):raise HTTPException(status.HTTP_404_NOT_FOUND,"Memory not found")
@app.post("/v1/documents")
async def index_document(body:DocumentRequest,user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    ids=await Retriever(session).index(body.source,body.title,body.content,user.id);return {"document_ids":ids}
@app.post("/v1/audio/transcriptions",response_model=ASRResponse)
async def transcribe(request:Request,file:UploadFile=File(...),language:str|None=None,user:User=Depends(current_user)):
    suffix=Path(file.filename or "audio.wav").suffix; data=await file.read()
    if len(data)>settings.max_upload_mb*1024*1024:raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,"Audio file too large")
    with NamedTemporaryFile(suffix=suffix,delete=False) as temporary:temporary.write(data); path=Path(temporary.name)
    try:
        recognizer=getattr(request.app.state,"asr",None) or SpeechRecognizer();request.app.state.asr=recognizer
        return await asyncio.to_thread(recognizer.transcribe,path,language)
    finally:path.unlink(missing_ok=True)
@app.post("/v1/audio/speech")
async def speech(body:TTSRequest,request:Request,background_tasks:BackgroundTasks,user:User=Depends(current_user)):
    output=Path("/tmp")/f"glm-{uuid4()}.wav"; synthesizer=getattr(request.app.state,"tts",None) or SpeechSynthesizer();request.app.state.tts=synthesizer
    try:
        await asyncio.to_thread(synthesizer.synthesize,body.text,output,body.language,None,body.emotion)
        background_tasks.add_task(output.unlink,missing_ok=True)
        return FileResponse(output,media_type="audio/wav",filename="speech.wav",background=background_tasks)
    except Exception:output.unlink(missing_ok=True);raise
@app.post("/v1/images/generations")
async def image_generation(body:ImageGenerationRequest,request:Request,background_tasks:BackgroundTasks,user:User=Depends(current_user)):
    output=Path("/tmp")/f"glm-{uuid4()}.png"; generator=getattr(request.app.state,"image_generator",None) or ImageGenerator();request.app.state.image_generator=generator
    try:
        await asyncio.to_thread(generator.generate,body.prompt,output,body.negative_prompt,body.width,body.height,body.steps,body.guidance,body.seed)
        background_tasks.add_task(output.unlink,missing_ok=True)
        return FileResponse(output,media_type="image/png",filename="generated.png",background=background_tasks)
    except Exception:output.unlink(missing_ok=True);raise
@app.post("/v1/videos/generations")
async def video_generation(body:VideoGenerationRequest,request:Request,background_tasks:BackgroundTasks,user:User=Depends(current_user)):
    output=Path("/tmp")/f"glm-{uuid4()}.mp4"; generator=getattr(request.app.state,"video_generator",None) or VideoGenerator();request.app.state.video_generator=generator
    try:
        await asyncio.to_thread(generator.generate,body.prompt,output,body.frames,body.fps,body.seed)
        background_tasks.add_task(output.unlink,missing_ok=True)
        return FileResponse(output,media_type="video/mp4",filename="generated.mp4",background=background_tasks)
    except Exception:output.unlink(missing_ok=True);raise
@app.post("/v1/images/understand")
async def understand_image(request:Request,file:UploadFile=File(...),prompt:str|None=None,user:User=Depends(current_user)):
    suffix=Path(file.filename or "image.png").suffix.lower()
    if suffix not in {".jpg",".jpeg",".png",".webp",".bmp"}:raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,"Supported image types: jpg, jpeg, png, webp, bmp")
    data=await file.read()
    if len(data)>settings.max_upload_mb*1024*1024:raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,"Image file too large")
    with NamedTemporaryFile(suffix=suffix,delete=False) as temporary:temporary.write(data);path=Path(temporary.name)
    try:
        understanding=getattr(request.app.state,"image_understanding",None) or ImageUnderstanding();request.app.state.image_understanding=understanding
        return {"description":await asyncio.to_thread(understanding.describe,path,prompt)}
    finally:path.unlink(missing_ok=True)
@app.get("/v1/platform/compute")
async def platform_compute(user:User=Depends(current_user)):
    devices=[]
    for index in range(torch.cuda.device_count()):
        properties=torch.cuda.get_device_properties(index)
        free,total=torch.cuda.mem_get_info(index)
        devices.append({"index":index,"name":properties.name,"memory_total_gb":round(total/1024**3,2),"memory_free_gb":round(free/1024**3,2),"compute_capability":f"{properties.major}.{properties.minor}"})
    return {"cuda_available":torch.cuda.is_available(),"gpu_count":len(devices),"devices":devices,"strategies":["single","deepspeed","fsdp"]}
@app.get("/v1/platform/datasets",response_model=list[DatasetResponse])
async def list_datasets(user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    return list(await session.scalars(select(DatasetAsset).where(DatasetAsset.user_id==user.id).order_by(DatasetAsset.created_at.desc())))
@app.post("/v1/platform/datasets",response_model=DatasetResponse,status_code=status.HTTP_201_CREATED)
async def upload_dataset(file:UploadFile=File(...),task:str="sft",user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    supported_tasks={"sft","dpo","asr","tts","emotion_recognition","emotion_generation","vision","video","motion","lipsync","gesture","memory"}
    if task not in supported_tasks:raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,"Unsupported training dataset task")
    if Path(file.filename or "").suffix.lower() not in {".jsonl",".json"}:raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,"Upload a JSONL dataset")
    content=await file.read()
    if len(content)>settings.max_upload_mb*1024*1024:raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,"Dataset file too large")
    try:
        lines=[line for line in content.decode("utf-8").splitlines() if line.strip()]
        for line in lines:json.loads(line)
    except (UnicodeDecodeError,json.JSONDecodeError):raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,"Dataset must contain one valid UTF-8 JSON object per line")
    if not lines:raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,"Dataset is empty")
    directory=training_manager.state_root/"uploads"/str(user.id);directory.mkdir(parents=True,exist_ok=True);dataset_id=str(uuid4());path=directory/f"{dataset_id}.jsonl";path.write_bytes(content)
    dataset=DatasetAsset(id=dataset_id,user_id=user.id,name=Path(file.filename or "dataset.jsonl").stem,path=str(path),format="jsonl",task=task,records=len(lines),size_bytes=len(content));session.add(dataset);await session.commit();await session.refresh(dataset);return dataset
@app.get("/v1/platform/runs",response_model=list[TrainingRunResponse])
async def list_training_runs(user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    return list(await session.scalars(select(TrainingRun).where(TrainingRun.user_id==user.id).order_by(TrainingRun.created_at.desc())))
@app.post("/v1/platform/runs",response_model=TrainingRunResponse,status_code=status.HTTP_201_CREATED)
async def launch_training_run(body:TrainingLaunchRequest,user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    dataset=await session.get(DatasetAsset,body.dataset_id)
    if dataset is None or dataset.user_id!=user.id:raise HTTPException(status.HTTP_404_NOT_FOUND,"Dataset not found")
    if dataset.task!=body.task:raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,"Dataset task does not match the requested training task")
    run_id=str(uuid4());output=training_manager.state_root/"checkpoints"/run_id;log=training_manager.state_root/"logs"/"training"/f"{run_id}.log"
    run=TrainingRun(id=run_id,user_id=user.id,name=body.name,task=body.task,status="queued",dataset_id=dataset.id,config_json=body.model_dump(),metrics_json={},output_dir=str(output),log_path=str(log));session.add(run);await session.commit();await session.refresh(run)
    try:return await training_manager.start(run.id)
    except ValueError as error:
        run.status="failed";run.metrics_json={"launch_error":str(error)};await session.commit();raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,str(error))
@app.get("/v1/platform/runs/{run_id}",response_model=TrainingRunResponse)
async def training_run(run_id:str,user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    run=await session.get(TrainingRun,run_id)
    if run is None or run.user_id!=user.id:raise HTTPException(status.HTTP_404_NOT_FOUND,"Training run not found")
    return run
@app.post("/v1/platform/runs/{run_id}/stop",response_model=TrainingRunResponse)
async def stop_training_run(run_id:str,user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    run=await session.get(TrainingRun,run_id)
    if run is None or run.user_id!=user.id:raise HTTPException(status.HTTP_404_NOT_FOUND,"Training run not found")
    if not await training_manager.stop(run_id):raise HTTPException(status.HTTP_409_CONFLICT,"Training run is not active")
    await session.refresh(run);return run
@app.get("/v1/platform/runs/{run_id}/logs",response_model=TrainingLogResponse)
async def training_logs(run_id:str,limit:int=300,user:User=Depends(current_user),session:AsyncSession=Depends(get_session)):
    run=await session.get(TrainingRun,run_id)
    if run is None or run.user_id!=user.id:raise HTTPException(status.HTTP_404_NOT_FOUND,"Training run not found")
    status_value,lines=await training_manager.tail(run_id,max(10,min(limit,1000)));return TrainingLogResponse(id=run_id,status=status_value,lines=lines)
