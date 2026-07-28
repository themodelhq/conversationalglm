from __future__ import annotations
import argparse
from training.datasets import download_manifest
def main():
 p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",default="data/raw");a=p.parse_args()
 for path in download_manifest(a.manifest,a.output):print(path)
if __name__=="__main__":main()
