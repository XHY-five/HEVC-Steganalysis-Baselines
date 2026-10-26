"""12-dimensional consistency distribution of motion vectors across PU groups."""
from collections import defaultdict
import numpy as np
from common.syntax import read_syntax,get_size


def mv(r):
    if 'mv_x' not in r or 'mv_y' not in r:return None
    return int(r['mv_x']),int(r['mv_y'])


def _consistency(blocks):
    # Count the blocks that share an identical MV with another small-block.
    total=len(blocks)
    if total<2:return 0
    vals=[mv(r) for r in blocks]
    seen=defaultdict(int)
    for v in vals:seen[v]+=1
    return sum(seen[v]>1 for v in vals)


def extract(row,config):
    records=read_syntax(row.get('mv_path') or row['original'])
    groups=defaultdict(list)
    for r in records:
        v=mv(r)
        if v is None:continue
        w,h=get_size(r)
        cat=0 if max(w,h)>=16 and min(w,h)==8 else 1 if (w,h)==(8,8) else 2 if max(w,h)==8 and min(w,h)==4 else 3 if (w,h)==(4,4) else None
        if cat is None:continue
        key=(int(r.get('frame',0)),int(r.get('x',0))//16,int(r.get('y',0))//16,cat)
        groups[key].append(r)
    # 2 + 3 + 2 + 3 distributions for the four sub-block families; two global summaries.
    dist=np.zeros(12,dtype=float);den=np.zeros(4,dtype=float)
    offsets=[0,2,5,7];widths=[2,3,2,3]
    for key,blocks in groups.items():
        cat=key[-1]; num=_consistency(blocks)
        if cat in (0,2):bucket=1 if num>=2 else 0
        else:bucket=2 if num>=4 else 1 if num>=3 else 0
        dist[offsets[cat]+bucket]+=1;den[cat]+=1
    for cat in range(4):
        off=offsets[cat];width=widths[cat]
        if den[cat]:dist[off:off+width]/=den[cat]
    # Additional consistency of horizontal and vertical neighboring blocks (12-D).
    mv_records=[r for r in records if mv(r) is not None]
    rmap={(int(r.get('frame',0)),int(r.get('x',0)),int(r.get('y',0))):mv(r) for r in mv_records}
    adjacent=[0,0];matches=[0,0]
    for r in mv_records:
        f=int(r.get('frame',0));x=int(r.get('x',0));y=int(r.get('y',0));w,h=get_size(r)
        for a,nkey in enumerate([(f,x+w,y),(f,x,y+h)]):
            if nkey in rmap:
                adjacent[a]+=1;matches[a]+=int(mv(r)==rmap[nkey])
    dist[10]=matches[0]/max(1,adjacent[0]);dist[11]=matches[1]/max(1,adjacent[1])
    if not groups and not any(adjacent):raise ValueError('No MV records with mv_x/mv_y and PU sizes')
    return {f'MVC{i+1:02d}':float(x) for i,x in enumerate(dist)}
