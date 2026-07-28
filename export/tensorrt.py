from __future__ import annotations
import argparse,shutil,subprocess
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--onnx",required=True);p.add_argument("--output",required=True);p.add_argument("--fp16",action="store_true");p.add_argument("--workspace-mib",type=int,default=4096);a=p.parse_args();binary=shutil.which("trtexec")
 if binary is None:raise SystemExit("TensorRT trtexec was not found. Install TensorRT and add trtexec to PATH.")
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);cmd=[binary,f"--onnx={a.onnx}",f"--saveEngine={a.output}",f"--memPoolSize=workspace:{a.workspace_mib}"]
 if a.fp16:cmd.append("--fp16")
 raise SystemExit(subprocess.call(cmd))
if __name__=="__main__":main()
