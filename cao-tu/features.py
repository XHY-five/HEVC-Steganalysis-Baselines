"""Eight CU types, 8 area ratios plus 2+2 PU/TU proportions."""
import numpy as np
from common.syntax import read_syntax,get_size


def classify(r):
    w,_=get_size(r);pu=int(r.get('partSize',r.get('pu',0)))
    td=int(r.get('tu_depth',r.get('tuDepth',r.get('tudepth',0))))
    if w>=64:return 1
    if w>=32:return 2
    if w>=16:return 3 if td<=0 else 4 if td==1 else 5
    if w>=8:return 8 if pu==3 else 6 if td==0 else 7
    return 8


def extract(row,config):
    records=read_syntax(row['original'])
    if any(not any(k in r for k in ('tu_depth','tuDepth','tudepth')) for r in records):
        raise ValueError('CU records require TU depth information for Cao-TU')
    area=np.zeros(8,dtype=float);number=np.zeros(8,dtype=float)
    for r in records:
        t=classify(r)-1;w,h=get_size(r)
        area[t]+=w*h;number[t]+=1
    if not area.sum():raise ValueError('No CU areas')
    ar=area/area.sum()
    den67=number[5]+number[6]
    den45=number[3]+number[4]
    f2=np.divide(number[[5,6]],den67) if den67 else np.zeros(2)
    f3=np.divide(number[[3,4]],den45) if den45 else np.zeros(2)
    vals=list(ar)+list(f2)+list(f3)
    return {f'AR{i+1}':float(x) for i,x in enumerate(vals[:8])} | {
        'P6':float(vals[8]),'P7':float(vals[9]),'P4':float(vals[10]),'P5':float(vals[11])}
