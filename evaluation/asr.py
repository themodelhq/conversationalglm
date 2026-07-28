from __future__ import annotations
import argparse,json
from pathlib import Path
from audio import SpeechRecognizer
from evaluation.metrics import word_error_rate
def main():
 p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--model",default="openai/whisper-small");p.add_argument("--output",default="evaluation/asr_results.json");a=p.parse_args();rows=[json.loads(x) for x in Path(a.data).read_text().splitlines()];recognizer=SpeechRecognizer(a.model);predictions=[recognizer.transcribe(row["audio_path"],row.get("language"))["text"] for row in rows];result={"wer":word_error_rate([x["transcript"] for x in rows],predictions),"predictions":predictions};Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({"wer":result["wer"]}))
if __name__=="__main__":main()
