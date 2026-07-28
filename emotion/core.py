from __future__ import annotations
from collections import Counter
import re

EMOTION_LABELS=("neutral","joy","sadness","anger","fear","surprise","disgust","calm")
LEXICON={"joy":{"happy","great","love","wonderful","excited","thanks"},"sadness":{"sad","sorry","miss","cry","loss","hurt"},"anger":{"angry","hate","furious","unfair","annoyed"},"fear":{"afraid","scared","anxious","worry","danger"},"surprise":{"wow","amazing","unexpected","really"},"disgust":{"disgusting","gross","awful"},"calm":{"relax","peaceful","calm","breathe"}}
def detect_text_emotion(text: str) -> dict[str,float]:
    words=set(re.findall(r"[\w']+",text.lower())); counts=Counter({label:len(words & terms) for label,terms in LEXICON.items()}); total=sum(counts.values())
    if not total: return {label:1.0 if label=="neutral" else 0.0 for label in EMOTION_LABELS}
    values={label:counts[label]/total for label in EMOTION_LABELS}; values["neutral"]=0.05; norm=sum(values.values()); return {k:v/norm for k,v in values.items()}
def expressive_prompt(emotion: str, text: str) -> str:
    if emotion not in EMOTION_LABELS: emotion="neutral"
    return f"[{emotion}] {text.strip()}"
