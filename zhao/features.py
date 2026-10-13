"""Zhao 39-dimensional IPM/SATD features from the supplied customized x265 candidate TXT."""
from collections import Counter
from functools import lru_cache
import re

LINE_PAT = re.compile(
    r"Frame=(?P<frame>\d+)\s+CTU=(?P<ctu>\d+)\s+x=(?P<x>\d+)\s+y=(?P<y>\d+)\s+"
    r"depth=(?P<depth>\d+)\s+partSize=(?P<part>\d+)\s+size=(?P<size>\d+x\d+)\s+"
    r"coverIPM=(?P<ipm>\d+)\s+coverBit=(?P<bit>[01])\s+MPM=\[(?P<mpm>[0-9,\s]*)\]\s+"
    r"modeCosts=\[(?P<costs>[0-9\.,\s]+)\]\s+SATD_sorted=\[(?P<satd>[0-9,\s]+)\]"
)

@lru_cache(maxsize=16)
def parse_ipm_txt(path):
    blocks={}
    with open(path, encoding='utf8',errors='ignore') as f:
        for line in f:
            m=LINE_PAT.search(line)
            if m is None:continue
            g=m.groupdict()
            frame=int(g['frame']);ctu=int(g['ctu']);x=int(g['x']);y=int(g['y'])
            blocks[(frame,ctu,x,y)]={
                'frame':frame, 'ctu':ctu, 'x':x,'y':y,
                'ipm':int(g['ipm']),
                'MPM':[int(k) for k in g['mpm'].split(',') if k.strip().isdigit()],
                'modeCosts':[float(k) for k in g['costs'].split(',') if k.strip()],
                'order':[int(k) for k in g['satd'].split(',') if k.strip().isdigit()]
            }
    if not blocks:
        raise ValueError(f'No candidate records found in {path}; Zhao needs customized x265 SATD TXT')
    return blocks


def extract(row, config):
    ori=parse_ipm_txt(str(row['original']))
    cal=parse_ipm_txt(str(row['calibrated']))
    frame=str(row.get('frame','')).strip()
    only=None if frame in ('','nan','None') else int(float(frame))
    max_modes=int(config.get('max_modes',35))
    beta=float(config.get('relative_cost_threshold',0.04))
    skip_pre1=bool(config.get('skip_mpm',True))
    ranked=Counter();dist=Counter();total=0
    for key,old in ori.items():
        if only is not None and key[0]!=only:continue
        new=cal.get(key)
        if new is None:continue
        if skip_pre1 and old['ipm'] in old['MPM']:continue
        order=new['order'];costs=new['modeCosts']
        if not order or len(costs)<max_modes:continue
        if old['ipm'] not in order or old['ipm']>=len(costs):continue
        rank=order.index(old['ipm'])+1
        dstar=min(costs)
        if dstar<=0:continue
        rel=abs(costs[old['ipm']]-dstar)/dstar
        category=1 if rel<=beta else 2 if rel<=2*beta else 3 if rel<=3*beta else 4
        ranked[rank]+=1;dist[category]+=1;total+=1
    if total==0:
        raise ValueError('No aligned candidate-mode and SATD records; check frame and candidate TXT pairs')
    features={f'F{i}':ranked[i]/total for i in range(1,max_modes+1)}
    features.update({f'G{i}':dist[i]/total for i in range(1,5)})
    return features
