"""Calculate decoder prediction error maps from reconstructed and predicted luma frames."""
import argparse
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--reconstructed',required=True)
    p.add_argument('--predicted',required=True);p.add_argument('--out',required=True)
    args=p.parse_args()
    y=np.load(args.reconstructed,allow_pickle=False).astype(np.float32)
    pred=np.load(args.predicted,allow_pickle=False).astype(np.float32)
    if y.shape!=pred.shape:raise ValueError('Expected same [T,H,W] shape')
    if y.ndim!=3:raise ValueError('Expected [T,H,W] frames')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    np.save(out,y-pred);print(f'Saved residual {y.shape} to {out}')

if __name__=='__main__':main()
