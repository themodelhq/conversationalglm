from __future__ import annotations
import re
from collections import Counter

def levenshtein(reference:list[str],hypothesis:list[str])->int:
 previous=list(range(len(hypothesis)+1))
 for i,token in enumerate(reference,1):
  current=[i]
  for j,candidate in enumerate(hypothesis,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(token!=candidate)))
  previous=current
 return previous[-1]
def word_error_rate(references:list[str],hypotheses:list[str])->float:
 if len(references)!=len(hypotheses):raise ValueError("references and hypotheses must have equal length")
 words=sum(len(re.findall(r"\w+",text.lower())) for text in references)
 return sum(levenshtein(re.findall(r"\w+",a.lower()),re.findall(r"\w+",b.lower())) for a,b in zip(references,hypotheses))/max(1,words)
def exact_match(references:list[str],hypotheses:list[str])->float:return sum(a.strip()==b.strip() for a,b in zip(references,hypotheses))/max(1,len(references))
