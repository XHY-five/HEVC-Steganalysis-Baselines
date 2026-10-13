"""Construct a paired dataset manifest from existing train/val/test label JSON maps."""
import argparse,json,csv
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',required=True);ap.add_argument('--test',required=True)
    ap.add_argument('--val');ap.add_argument('--index',required=True)
    ap.add_argument('--output',default='manifest.csv')
    ap.add_argument('--payload',type=float,required=True);ap.add_argument('--tar',required=True)
    args=ap.parse_args()
    index_path=Path(args.index).resolve()
    with index_path.open(encoding='utf-8-sig',newline='') as f:
        index={row['sample_id']:row for row in csv.DictReader(f)}
    result=[]
    for split,path in [('train',args.train),('val',args.val),('test',args.test)]:
        if not path:continue
        labels=json.loads(Path(path).read_text(encoding='utf-8'))
        for sid,label in labels.items():
            if sid not in index:raise ValueError(f'Could not find sample_id in index: {sid}')
            row=index[sid].copy();row['label']=int(label);row['sample_id']=sid
            row['split']=split;row['payload']=args.payload;row['tar']=args.tar
            result.append(row)
    columns=['sample_id','label','split','tar','payload']+sorted({k for x in result for k in x.keys()}-{'sample_id','label','split','tar','payload'})
    out=Path(args.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(result)
    print(f'Wrote {len(result)} samples to {out}')

if __name__=='__main__':main()
