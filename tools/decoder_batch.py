"""Invoke an external instrumented decoder for a batch of bitstreams."""
import argparse,csv,subprocess,shlex
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',required=True,help='CSV with bitstream and output_log columns')
    p.add_argument('--command',required=True,help='Shell-like command with {input} and {log}, e.g. decoder -b {input} -o {log}')
    args=p.parse_args()
    with open(args.index,encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            target=Path(row['output_log']);target.parent.mkdir(parents=True,exist_ok=True)
            cmd=args.command.replace('{input}',row['bitstream']).replace('{log}',str(target))
            with target.open('w',encoding='utf-8') as dst:
                subprocess.run(shlex.split(cmd),check=True,stdout=dst,stderr=subprocess.STDOUT)
            print(f'Wrote {target}')

if __name__=='__main__':main()
