from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--gpus",type=int,default=1);p.add_argument("--nodes",type=int,default=1);p.add_argument("--deepspeed");p.add_argument("--fsdp",action="store_true");a=p.parse_args();cmd=["accelerate","launch","--num_processes",str(a.gpus),"--num_machines",str(a.nodes)]
 if a.deepspeed:cmd.extend(["--use_deepspeed","--deepspeed_config_file",a.deepspeed])
 if a.fsdp:cmd.append("--use_fsdp")
 cmd.extend(["-m","training.train","--config",a.config]);raise SystemExit(subprocess.call(cmd))
if __name__=="__main__":main()
