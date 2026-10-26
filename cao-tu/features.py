"""Cao-TU CU/PU/TU partition statistics from the same decoded HM TXT format.

The record must expose a true TU-depth field. CU depth and IPM alone do not
contain the TU decisions necessary for this cross-domain feature.
"""
from collections import Counter
import numpy as np
from common.decoded import records
from common.syntax import get_size


def cu_category(r):
    w,h=get_size(r)
    d=int(r['tu_depth'])
    pu=int(r['partSize'])
    # Eight structural families (CU size / PU / TU decisions).
    if min(w,h)>=64:return 0
    if min(w,h)>=32:return 1
    if min(w,h)>=16:return 2 if d==0 else 3 if d==1 else 4
    if min(w,h)>=8:return 7 if pu==3 else 5 if d==0 else 6
    return 7


def extract(row, config):
    src=records(row,required=('depth','partSize','tu_depth'))
    areas=np.zeros(8,float);counts=np.zeros(8,float)
    for r in src:
        code=cu_category(r)
        w,h=get_size(r)
        areas[code]+=w*h
        counts[code]+=1
    if areas.sum()<=0:raise ValueError('No valid CU/PU/TU records')
    ar=areas/areas.sum()
    denom_a=counts[5]+counts[6]
    denom_b=counts[3]+counts[4]
    extra=[counts[5]/denom_a if denom_a else 0.,
           counts[6]/denom_a if denom_a else 0.,
           counts[3]/denom_b if denom_b else 0.,
           counts[4]/denom_b if denom_b else 0.]
    return {**{f'AR{i+1}':float(v) for i,v in enumerate(ar)},
            'P6':extra[0],'P7':extra[1], 'P4':extra[2],'P5':extra[3]}
