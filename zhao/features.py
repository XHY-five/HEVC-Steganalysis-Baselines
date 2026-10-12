"""39-dimensional IPM calibration / relative SATD statistics."""
from collections import Counter
import numpy as np
from common.syntax import read_syntax,pair_by_location


def extract(row,config):
    original=read_syntax(row['original']);cal=read_syntax(row['calibrated'])
    rank=Counter();deviation=Counter();n=0
    beta=float(config.get('relative_cost_threshold',0.04))
    for a,b in pair_by_location(original,cal):
        ipm=int(a.get('ipm',-1))
        if ipm<0 or ipm>=35:continue
        mpm=a.get('MPM',a.get('mpm',[]))
        if bool(config.get('skip_mpm',True)) and ipm in mpm:continue
        order=b.get('order',b.get('satd_order',[]))
        costs=b.get('costs',b.get('modeCosts',[]))
        if not order or not costs or len(costs)<35:continue
        order=list(map(int,order));costs=list(map(float,costs))
        if ipm not in order or len(costs)<=ipm:continue
        r=order.index(ipm)+1
        mincost=min(costs)
        if mincost<=0:continue
        dev=abs(costs[ipm]-mincost)/mincost
        bucket=min(4,int(dev/beta)+1) if beta>0 else 4
        rank[r]+=1;deviation[bucket]+=1;n+=1
    if n==0:raise ValueError('No calibrated IPM/SATD pairs with valid mode costs')
    return {**{f'F{i:02d}':rank[i]/n for i in range(1,36)},
            **{f'G{i}':deviation[i]/n for i in range(1,5)}}
