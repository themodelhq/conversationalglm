from __future__ import annotations
import re
from typing import Iterable
PATTERNS=(re.compile(r"\bmy name is ([A-Za-z][A-Za-z '\-]{1,60})",re.I),re.compile(r"\bI (?:live|am based) in ([A-Za-z][A-Za-z ,.'\-]{1,80})",re.I),re.compile(r"\bI (?:like|love|prefer) ([^.?!]{2,100})",re.I),re.compile(r"\bremember (?:that )?([^.?!]{2,180})",re.I))
def extract_memories(text: str) -> list[tuple[str,float]]:
    found=[]
    for pattern in PATTERNS:
        for match in pattern.finditer(text): found.append((match.group(0).strip(),.75 if "remember" in match.group(0).lower() else .55))
    return found
