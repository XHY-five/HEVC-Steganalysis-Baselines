"""Zhai 12-dimensional motion-vector consistency, reading HM-decoder TXT directly.

These are true MV comparisons. Never substitute IPM values for mv_x/mv_y.
"""
from collections import defaultdict
import numpy as np
from common.decoded import records
from common.syntax import get_size


def vector(r):return (int(r['mv_x']),int(r['mv_y']))


def _consistent_count(items):
    count=defaultdict(int)
    for r in items:count[vector(r)]+=1
    return sum(count[vector(r)]>1 for r in items)


def extract(row,config):
    src=records(row,required=('mv_x','mv_y','partSize','depth'))
    groups=defaultdict(list)
    for r in src:
        w,h=get_size(r)
        cat=(0 if max(w,h)>=16 and min(w,h)==8 else
             1 if (w,h)==(8,8) else
             2 if max(w,h)==8 and min(w,h)==4 else
             3 if (w,h)==(4,4) else None)
        if cat is None:continue
        key=(int(r['frame']),int(r['x'])//16,int(r['y'])//16,cat)
        groups[key].append(r)
    result=np.zeros(12,float);den=np.zeros(4,float)
    offsets=(0,2,5,7);lengths=(2,3,2,3)
    for key,items in groups.items():
        cat=key[-1];matches=_consistent_count(items)
        bucket=(int(matches>=2) if cat in (0,2)
                else 2 if matches>=4 else 1 if matches>=3 else 0)
        result[offsets[cat]+bucket]+=1
        den[cat]+=1
    for cat in range(4):
        if den[cat]:result[offsets[cat]:offsets[cat]+lengths[cat]]/=den[cat]
    points={(int(r['frame']),int(r['x']),int(r['y'])):vector(r) for r in src}
    totals=[0,0];same=[0,0]
    for r in src:
        f=int(r['frame']);x=int(r['x']);y=int(r['y']);w,h=get_size(r)
        for j,k in enumerate(((f,x+w,y),(f,x,y+h))):
            if k in points:
                totals[j]+=1
                same[j]+=int(points[k]==vector(r))
    result[10]=same[0]/max(1,totals[0]);result[11]=same[1]/max(1,totals[1])
    if not groups and not any(totals):
        raise ValueError('MV records have no supported PU partition size for 12-D consistency features')
    return {f'MVC{i+1:02d}':float(v) for i,v in enumerate(result)}
