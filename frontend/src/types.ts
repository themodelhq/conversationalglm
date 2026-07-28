export type Role = 'system' | 'user' | 'assistant' | 'tool';
export type TrainingTask = 'sft' | 'dpo' | 'asr' | 'tts' | 'emotion_recognition' | 'emotion_generation' | 'vision' | 'video' | 'motion' | 'lipsync' | 'gesture' | 'memory';
export type RunStatus = 'queued' | 'running' | 'stopping' | 'completed' | 'failed' | 'cancelled' | 'interrupted';

export interface Message { role: Role; content: string; }
export interface ChatResponse { conversation_id: string; content: string; sources: Array<{title:string; source:string; content:string; score:number}>; memories: Array<{content:string; score:number}>; tool_calls: Array<{name:string; result:unknown}>; }
export interface DatasetAsset { id:string; name:string; format:string; task:TrainingTask; records:number; size_bytes:number; created_at:string; }
export interface ComputeDevice { index:number; name:string; memory_total_gb:number; memory_free_gb:number; compute_capability:string; }
export interface ComputeInfo { cuda_available:boolean; gpu_count:number; devices:ComputeDevice[]; strategies:string[]; }
export interface TrainingConfig { name:string; dataset_id:string; task:TrainingTask; model_size:'starter'|'standard'|'advanced'; epochs:number; batch_size:number; gradient_accumulation_steps:number; learning_rate:number; max_length:number; mixed_precision:'no'|'fp16'|'bf16'; strategy:'single'|'deepspeed'|'fsdp'; gpu_count:number; node_count:number; machine_rank:number; main_process_ip:string; main_process_port:number; tracking:'none'|'tensorboard'|'wandb'; warmup_ratio:number; weight_decay:number; save_steps:number; eval_steps:number; }
export interface TrainingRun { id:string; name:string; task:TrainingTask; status:RunStatus; dataset_id:string; config_json:TrainingConfig; metrics_json:Record<string, number | string>; output_dir:string; process_id:number | null; return_code:number | null; created_at:string; started_at:string | null; finished_at:string | null; }
export interface TrainingLog { id:string; status:RunStatus; lines:string[]; }
