from __future__ import annotations
from collections import defaultdict
import random
from typing import Iterable

def balanced_indices(labels:Iterable[str],seed:int=42)->list[int]:
    groups:dict[str,list[int]]=defaultdict(list)
    for index,label in enumerate(labels):groups[str(label)].append(index)
    if not groups:return []
    rng=random.Random(seed);limit=max(len(indices) for indices in groups.values());output=[]
    for indices in groups.values():output.extend(indices);output.extend(rng.choice(indices) for _ in range(limit-len(indices)))
    rng.shuffle(output);return output
