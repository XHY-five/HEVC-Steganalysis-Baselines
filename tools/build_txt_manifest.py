"""Pair the four original HM decoded log sets into one manifest, without random repartitioning."""
import argparse
import csv
import json
from pathlib import Path

KIND={'cover':'cover_ipm','cover_cal':'cover_cal_ipm','stego':'steg_ipm','stego_cal':'steg_cal_ipm'}

def scan(directory,suffix):
    directory=Path(directory)
    result={}
    for p in sorted(directory.rglob('*_'+suffix+'.txt')):
        segment=p.parent.name
        if segment in result:raise ValueError(f'Duplicate TXT for {segment} under {directory}')
        result[segment]=p.resolve()
    return result

def main():
    p=argparse.ArgumentParser(description='Pair decoder-produced TXT using segment names')
    for k in KIND:
        p.add_argument('--'+k.replace('_','-'),required=True,help=f'Folder containing */*_{KIND[k]}.txt')
    p.add_argument('--out',required=True)
    p.add_argument('--tar',required=True)
    p.add_argument('--payload',type=float,required=True)
    p.add_argument('--splits',help='Existing CSV with sample_id and split columns, keyed by segment or segment_stego')
    a=p.parse_args()
    paths={k:scan(getattr(a,k),suffix) for k,suffix in KIND.items()}
    entries={}
    if a.splits:
        with open(a.splits,encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                if row['sample_id'] in entries:raise ValueError('Duplicate sample_id in split CSV')
                entries[row['sample_id']]=row['split']
    rows=[]
    for group,cal,label in [('cover','cover_cal',0),('stego','stego_cal',1)]:
        common=sorted(set(paths[group])&set(paths[cal]))
        for segment in common:
            sid=segment+('_stego' if label else '')
            split=entries.get(sid,'unassigned')
            rows.append(dict(sample_id=sid,label=label,split=split,tar=a.tar,payload=a.payload,
                             original=str(paths[group][segment]),calibrated=str(paths[cal][segment])))
    out=Path(a.out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=('sample_id','label','split','tar','payload','original','calibrated'))
        w.writeheader();w.writerows(rows)
    print(f'Wrote {len(rows)} pairs to {out}; train/val/test supplied via --splits; unassigned rows cannot be trained')
    missing={k:len(v) for k,v in paths.items()};print('TXT counts:',missing)
    if not rows:raise RuntimeError('No original/calibrated TXT pairs found; verify folder names and file suffixes')

if __name__=='__main__':main()
