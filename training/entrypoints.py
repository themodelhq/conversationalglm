from __future__ import annotations
import argparse,sys
from training.multimodal import run as multimodal_run
def language(config:str)->None:
 from training.train import main
 sys.argv=[sys.argv[0],"--config",config];main()
def multimodal(task:str)->None:multimodal_run(task)
