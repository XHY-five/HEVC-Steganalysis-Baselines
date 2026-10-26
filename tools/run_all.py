"""Run eight baseline evaluations sequentially on the same manifest."""
import argparse,subprocess,sys
from pathlib import Path

METHODS=('cao','shen','zhao','cao264','dai-pu','dai','cao-tu','zhai')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--tar');p.add_argument('--payload',type=float)
    p.add_argument('--out',default='results_all');p.add_argument('--methods',nargs='+',default=list(METHODS))
    p.add_argument('--epochs',type=int,default=None)
    p.add_argument('--zhao-manifest',help='Separate candidate-SATD TXT manifest for Zhao; labels and splits must match common manifest')
    args=p.parse_args();root=Path(__file__).resolve().parents[1]
    for method in args.methods:
        if method not in METHODS:raise ValueError(f'Unknown method: {method}')
        input_manifest=(args.zhao_manifest if method=='zhao' and args.zhao_manifest else args.manifest)
        cmd=[sys.executable,str(root/method/'run.py'),'--manifest',input_manifest,'--out',str(Path(args.out)/method)]
        if args.tar:cmd+=['--tar',args.tar]
        if args.payload is not None:cmd+=['--payload',str(args.payload)]
        if args.epochs is not None and method in ('dai','dai-pu'):cmd+=['--epochs',str(args.epochs)]
        print('RUN:', ' '.join(cmd),flush=True)
        subprocess.run(cmd,check=True)

if __name__=='__main__':main()
