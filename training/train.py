from __future__ import annotations
import argparse,copy,json,math,os,random
from pathlib import Path
import numpy as np
import torch
import yaml
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from models import ConversationalGLM, ConversationTokenizer, GLMConfig
from models.glm import GLMOutput
from safetensors.torch import load_file
from training.data import JsonlDataset, collate_dpo, collate_sft

def seed_everything(seed:int):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def sequence_logprob(model,ids,pad_id:int):
    mask=ids.ne(pad_id); logits=model(ids,mask).logits[:,:-1]; labels=ids[:,1:]; token=torch.gather(torch.log_softmax(logits,-1),-1,labels.unsqueeze(-1)).squeeze(-1); return (token*mask[:,1:]).sum(-1)
def evaluate(model,loader,accelerator,task,pad_id,reference=None):
    model.eval();values=[]
    with torch.no_grad():
        for batch in loader:
            if task=="dpo":
                chosen,rejected=batch["chosen"],batch["rejected"]; policy=sequence_logprob(model,chosen,pad_id)-sequence_logprob(model,rejected,pad_id); ref=sequence_logprob(reference,chosen,pad_id)-sequence_logprob(reference,rejected,pad_id); loss=-torch.nn.functional.logsigmoid(.1*(policy-ref)).mean()
            else:loss=model(**batch).loss
            values.extend(accelerator.gather_for_metrics(loss.detach().reshape(1)).float().cpu().tolist())
    model.train();return float(np.mean(values)) if values else float("inf")
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--resume",default=None);args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text());run,model_cfg,data_cfg,train_cfg=cfg["run"],cfg["model"],cfg["data"],cfg["training"];seed_everything(run.get("seed",42))
    mixed=train_cfg.get("mixed_precision","no");mixed="no" if mixed not in {"fp16","bf16"} else mixed;tracking=train_cfg.get("tracking","tensorboard");log_with=tracking if tracking in {"tensorboard","wandb"} else None;accelerator=Accelerator(gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],mixed_precision=mixed,log_with=log_with,project_dir=run["output_dir"])
    output=Path(run["output_dir"]);output.mkdir(parents=True,exist_ok=True)
    if log_with:accelerator.init_trackers(run["name"],config=cfg)
    tokenizer=ConversationTokenizer.load(data_cfg["tokenizer"]);config=GLMConfig.from_file(model_cfg["config"]);config.vocab_size=len(tokenizer);config.pad_token_id=tokenizer.pad_token_id;config.bos_token_id=tokenizer.bos_token_id;config.eos_token_id=tokenizer.eos_token_id;config.gradient_checkpointing=train_cfg.get("gradient_checkpointing",False)
    model=ConversationalGLM(config)
    if model_cfg.get("init_from"):
        source=Path(model_cfg["init_from"]);weights=source if source.is_file() else source/"model.safetensors";model.load_state_dict(load_file(str(weights)),strict=False)
    task=train_cfg.get("task","sft");train_set=JsonlDataset(data_cfg["train"],tokenizer,config.max_position_embeddings,task);valid_set=JsonlDataset(data_cfg["validation"],tokenizer,config.max_position_embeddings,task)
    collate=(lambda batch:collate_dpo(batch,tokenizer.pad_token_id)) if task=="dpo" else (lambda batch:collate_sft(batch,tokenizer.pad_token_id));loader=DataLoader(train_set,batch_size=train_cfg["batch_size"],shuffle=True,num_workers=train_cfg.get("num_workers",0),pin_memory=True,collate_fn=collate);validation=DataLoader(valid_set,batch_size=train_cfg["batch_size"],shuffle=False,num_workers=train_cfg.get("num_workers",0),pin_memory=True,collate_fn=collate)
    optimizer=AdamW(model.parameters(),lr=float(train_cfg["learning_rate"]),weight_decay=float(train_cfg["weight_decay"]));steps_per_epoch=math.ceil(len(loader)/train_cfg["gradient_accumulation_steps"]);total=train_cfg.get("max_steps",-1);total=total if total>0 else steps_per_epoch*train_cfg["epochs"];scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min(1,(step+1)/max(1,int(total*train_cfg.get("warmup_ratio",.03))))*max(0.,1-(step+1)/total))
    reference=None
    if task=="dpo":reference=copy.deepcopy(model).eval();reference.requires_grad_(False)
    if reference is None:model,optimizer,loader,validation,scheduler=accelerator.prepare(model,optimizer,loader,validation,scheduler)
    else:model,reference,optimizer,loader,validation,scheduler=accelerator.prepare(model,reference,optimizer,loader,validation,scheduler)
    start=0;resume=args.resume or run.get("resume_from")
    if resume:accelerator.load_state(resume);start=int(Path(resume).name.split("-")[-1]) if Path(resume).name.startswith("step-") else 0
    best=float("inf");stale=0;step=start
    for epoch in range(train_cfg["epochs"]):
        for batch in loader:
            with accelerator.accumulate(model):
                if task=="dpo":
                    policy=sequence_logprob(model,batch["chosen"],tokenizer.pad_token_id)-sequence_logprob(model,batch["rejected"],tokenizer.pad_token_id);ref=sequence_logprob(reference,batch["chosen"],tokenizer.pad_token_id)-sequence_logprob(reference,batch["rejected"],tokenizer.pad_token_id);loss=-torch.nn.functional.logsigmoid(float(train_cfg.get("beta",.1))*(policy-ref)).mean()
                else:loss=model(**batch).loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:accelerator.clip_grad_norm_(model.parameters(),train_cfg["max_grad_norm"])
                optimizer.step();scheduler.step();optimizer.zero_grad()
            if accelerator.sync_gradients:
                step+=1
                if step%train_cfg["logging_steps"]==0:accelerator.log({"train/loss":loss.detach().float().item(),"train/lr":scheduler.get_last_lr()[0]},step=step)
                if step%train_cfg["eval_steps"]==0:
                    value=evaluate(model,validation,accelerator,task,tokenizer.pad_token_id,reference);accelerator.log({"validation/loss":value},step=step)
                    if value<best:
                        best=value;stale=0;accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            accelerator.unwrap_model(model).save_pretrained(output/"best");tokenizer.save_pretrained(output/"best")
                    else:stale+=1
                    if stale>=train_cfg.get("early_stopping_patience",999):break
                if step%train_cfg["save_steps"]==0:accelerator.save_state(output/f"step-{step}")
                if step>=total:break
        if stale>=train_cfg.get("early_stopping_patience",999) or step>=total:break
    accelerator.wait_for_everyone();
    if accelerator.is_main_process:accelerator.unwrap_model(model).save_pretrained(output);tokenizer.save_pretrained(output);(output/"training_summary.json").write_text(json.dumps({"steps":step,"best_validation_loss":best,"task":task},indent=2))
    accelerator.end_training()
if __name__=="__main__":main()
