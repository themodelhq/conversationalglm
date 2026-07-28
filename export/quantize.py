from __future__ import annotations
import argparse
from pathlib import Path
from onnxruntime.quantization import QuantType, quantize_dynamic
def main():
 p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True);p.add_argument("--weight-type",choices=["qint8","quint8"],default="qint8");a=p.parse_args();Path(a.output).parent.mkdir(parents=True,exist_ok=True);quantize_dynamic(a.input,a.output,weight_type=QuantType.QInt8 if a.weight_type=="qint8" else QuantType.QUInt8,per_channel=True)
if __name__=="__main__":main()
