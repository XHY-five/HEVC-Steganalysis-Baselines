"""Convert instrumented HEVC decoder syntax logs to PU-map tensor files."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from common.syntax import read_syntax,block_grid
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--log',required=True);p.add_argument('--out',required=True)
    p.add_argument('--block',type=int,default=4)
    args=p.parse_args()
    arr=block_grid(read_syntax(args.log),keys=('partSize',),block=args.block)
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    np.save(path,arr)
    print(f'{arr.shape}: {path}')

if __name__=='__main__':main()
