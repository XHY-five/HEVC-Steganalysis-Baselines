"""11-dimensional partition-depth calibration differences."""
from collections import defaultdict
import numpy as np
from common.syntax import read_syntax,get_size


def depth_code(r):
    w,h=get_size(r)
    if w>=64:return 0
    if w>=32:return 1
    if w>=16:return 2
    return 4 if int(r.get('partSize',0))==3 else 3


def matrices(records):
    out=defaultdict(lambda: (np.full((16,16),-1,dtype=np.int16),np.full((16,16),-1,dtype=np.int16)))
    for r in records:
        f=int(r.get('frame',0));ctu=int(r.get('ctu',0))
        d,im=out[(f,ctu)]
        w,h=get_size(r)
        x=int(r.get('x',0))%64//4;y=int(r.get('y',0))%64//4
        wi=max(1,w//4);hi=max(1,h//4)
        d[y:y+hi,x:x+wi]=depth_code(r)
        im[y:y+hi,x:x+wi]=int(r.get('ipm',-1))
    return out


def extract(row,config):
    a=matrices(read_syntax(row['original']))
    b=matrices(read_syntax(row['calibrated']))
    diff=[];C4=C4p=0;total=0
    N=np.zeros(5,dtype=float);N2=np.zeros(5,dtype=float);unstable=np.zeros(5,dtype=float)
    for key in a.keys() & b.keys():
        da,ia=a[key];db,ib=b[key]
        mask=(da>=0)&(db>=0)
        if not mask.any():continue
        diff.append(float(np.abs(da[mask]-db[mask]).sum()))
        C4+=int(np.count_nonzero((da==4)&(db!=4)&mask))
        C4p+=int(np.count_nonzero((da!=4)&(db==4)&mask))
        total+=int(mask.sum())
        changes=(da!=db)|(ia!=ib)
        for dep in range(5):
            N[dep]+=int(np.count_nonzero((da==dep)&mask))
            N2[dep]+=int(np.count_nonzero((db==dep)&mask))
            unstable[dep]+=int(np.count_nonzero((da==dep)&changes&mask))
    if total==0:raise ValueError('No matched CU-depth records')
    names=['F0_DPSavg','F1_CP4','F1_CP4p',
           'F2_DDP1','F2_DDP2','F2_DDP3','F2_DDP4',
           'F3_UP1','F3_UP2','F3_UP3','F3_UP4']
    result=[np.mean(diff),C4/total,C4p/total]
    for i in range(1,5):result.append(0 if N[i]==0 else ((N[i]-N2[i])/N[i]))
    for i in range(1,5):result.append(0 if N[i]==0 else (unstable[i]/N[i]))
    return dict(zip(names,map(float,result)))
