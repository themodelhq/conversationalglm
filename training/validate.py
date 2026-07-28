from __future__ import annotations
import argparse,json
from pathlib import Path
REQUIRED={"sft":set(),"dpo":{"prompt","chosen","rejected"},"asr":{"audio_path","transcript"},"tts":{"audio_path","transcript"},"vision":{"image_path","text"},"video":{"video_path","text"}}
def main():
    parser=argparse.ArgumentParser();parser.add_argument("path");parser.add_argument("--task",default="sft",choices=REQUIRED);args=parser.parse_args();bad=[];count=0
    with Path(args.path).open(encoding="utf-8") as stream:
        for number,line in enumerate(stream,1):
            try:row=json.loads(line);missing=REQUIRED[args.task]-row.keys(); assert not missing
            except (json.JSONDecodeError,AssertionError):bad.append(number)
            else:count+=1
    print(json.dumps({"valid":count,"invalid":len(bad),"invalid_lines":bad[:100]}));raise SystemExit(1 if bad else 0)
if __name__=="__main__":main()
