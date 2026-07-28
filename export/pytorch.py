from __future__ import annotations
import argparse
from pathlib import Path
from models import ConversationalGLM, ConversationTokenizer
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--output",required=True);a=p.parse_args();model=ConversationalGLM.from_pretrained(a.model);out=Path(a.output);model.save_pretrained(out);ConversationTokenizer.load(a.model).save_pretrained(out)
if __name__=="__main__":main()
