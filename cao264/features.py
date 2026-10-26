"""SUPERB-style PEB differences & IPM transitions from paired HM decoder TXT.

A true residual/PEB field is mandatory in the decoded records. Original and
calibrated PEB maps replace the separate NPZ input of the older implementation.
"""
import numpy as np
from common.decoded import residual_abs_maps, records, make_map


def _transitions(arr,dx,dy,T):
    src=np.clip(np.rint(arr),0,T).astype(np.int64)
    h,w=src.shape
    ya=max(0,-dy);yb=max(0,dy);xa=max(0,-dx);xb=max(0,dx)
    ih=h-abs(dy);iw=w-abs(dx)
    if ih<=0 or iw<=0:return np.zeros((T+1)**2,float)
    p=src[ya:ya+ih,xa:xa+iw]
    q=src[yb:yb+ih,xb:xb+iw]
    counts=np.bincount((p*(T+1)+q).ravel(),minlength=(T+1)**2).astype(float).reshape(T+1,T+1)
    return np.divide(counts,counts.sum(axis=1,keepdims=True),out=np.zeros_like(counts),where=counts.sum(axis=1,keepdims=True)>0).ravel()


def _mode_group(mode):
    """Compact 9-category directional coding of 35 HEVC IPM indices."""
    mode=int(mode)
    if mode in (0,1):return mode
    return min(8,2+int((mode-2)*7/33))


def extract(row,config):
    R,rm,rf=residual_abs_maps(row,'original')
    C,cm,cf=residual_abs_maps(row,'calibrated')
    if rf!=cf or R.shape!=C.shape:
        raise ValueError('The original and calibrated PEB TXT do not cover the same frame size/IDs')
    if not (rm.all() and cm.all()):
        raise ValueError('Incomplete original/calibrated PEB block records')
    if len(rf)<2:
        raise ValueError('SUPERB temporal features need a clip with >=2 frames in TXT; do not select a single-frame row')
    T=int(config.get('truncation',4))
    if T<1:raise ValueError('truncation must be positive')
    maps=[np.abs(R-C),np.abs(R[1:]-R[:-1]),
          np.abs(R[:,1:,:]-R[:,:-1,:]),np.abs(R[:,:,1:]-R[:,:,:-1]),
          np.abs(R[:,1:,1:]-R[:,:-1,:-1]),
          np.abs(R[:,1:,:-1]-R[:,:-1,1:])]
    feats=[]
    for chunk in maps:
        if any(d==0 for d in chunk.shape):
            raise ValueError('PEB grid too small for spatial/temporal transitions')
        for dx,dy in ((-1,-1),(0,-1),(1,-1),(-1,0)):
            feats.extend(np.mean([_transitions(a,dx,dy,T) for a in chunk],axis=0).tolist())
    oa=records(row,'original',required=('ipm',))
    ca=records(row,'calibrated',required=('ipm',))
    loc=lambda r:(int(r['frame']),int(r['x']),int(r['y']))
    changed={loc(r):int(r['ipm']) for r in ca}
    for k in range(9):
        values=[float(_mode_group(changed[loc(r)])==k)
                for r in oa if _mode_group(r['ipm'])==k and loc(r) in changed]
        feats.append(float(np.mean(values)) if values else 0.)
    expected=6*4*(T+1)**2+9
    if len(feats)!=expected:raise AssertionError(f'SUPERB dimensionality {len(feats)}!={expected}')
    return {f'SUPERB_{j:04d}':float(v) for j,v in enumerate(feats)}
