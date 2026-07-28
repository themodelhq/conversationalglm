from __future__ import annotations
import argparse
from pathlib import Path
from models.configuration_glm import GLMConfig
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--output",required=True);a=p.parse_args();cfg=GLMConfig.from_file(Path(a.model)/"config.json")
 raise SystemExit("GGUF export is intentionally unavailable for ConversationalGLM: GGUF runtimes require a registered architecture and tokenizer conversion. Export ONNX or TensorRT for this architecture, or implement a llama.cpp architecture binding before enabling GGUF.")
if __name__=="__main__":main()
