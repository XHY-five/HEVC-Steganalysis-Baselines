"""SUPERB-inspired subtractive prediction-error features and IPM transitions."""
import numpy as np
from common.data import read_array


def pool_block_abs(x, block_size):
    if x.ndim==5:
        return np.mean(np.abs(x),axis=(-2,-1))
    if x.ndim==3:
        t,h,w=x.shape
        h2=h//block_size;w2=w//block_size
        if not h2 or not w2:raise ValueError('Residual frames are smaller than a block')
        x=x[:,:h2*block_size,:w2*block_size]
        return np.abs(x).reshape(t,h2,block_size,w2,block_size).mean(axis=(2,4))
    raise ValueError('Expected residual array shaped [T,H,W] or [T,Hb,Wb,b,b]')


def transitions(A,dx,dy,T):
    if A.ndim!=2:return np.zeros((T+1)*(T+1),dtype=np.float32)
    a=np.clip(np.rint(A),0,T).astype(np.int64)
    s1=a[max(0,-dy):a.shape[0]-max(0,dy),max(0,-dx):a.shape[1]-max(0,dx)]
    s2=a[max(0,dy):a.shape[0]-max(0,-dy),max(0,dx):a.shape[1]-max(0,-dx)]
    if s1.size==0:return np.zeros((T+1)*(T+1),dtype=np.float32)
    h=np.bincount((s1*(T+1)+s2).ravel(),minlength=(T+1)**2).reshape(T+1,T+1).astype(float)
    norm=h.sum(axis=1,keepdims=True)
    h=np.divide(h,norm,out=np.zeros_like(h),where=norm>0)
    return h.ravel()


def load_residual_fields(path):
    with np.load(path,allow_pickle=False) as z:
        if 'residual' not in z:raise ValueError('Residual .npz needs residual key')
        residual=np.asarray(z['residual'],dtype=np.float32)
        required={'reestimated_residual','ipm','reestimated_ipm'}
        absent=required-set(z.files)
        if absent:raise ValueError(f'Residual .npz missing keys: {sorted(absent)}')
        other=np.asarray(z['reestimated_residual'],dtype=np.float32)
        ipm=np.asarray(z['ipm'],dtype=int)
        ipm_other=np.asarray(z['reestimated_ipm'],dtype=int)
    return residual,other,ipm,ipm_other


def extract(row,config):
    path=row.get('residual_path') or row.get('data_path','')
    if not path:raise ValueError('Manifest needs residual_path / data_path pointing to .npz residual blocks')
    R,O,ipm,ipm2=load_residual_fields(path)
    if R.shape!=O.shape:raise ValueError('Original and reestimated residual arrays must match')
    block=int(config.get('block_size',4));T=int(config.get('truncation',4))
    orig=pool_block_abs(R,block);reest=pool_block_abs(O,block)
    if orig.shape[0]<2:raise ValueError('At least two residual frames needed')
    # Differential maps associated with re-estimated prediction, temporal
    # neighbors and spatially adjacent PEBs.
    kinds=[]
    kinds.append(np.abs(orig-reest))
    tmp=np.abs(orig[1:]-orig[:-1]);kinds.append(tmp)
    up=np.abs(orig[:,1:,:]-orig[:,:-1,:]);right=np.abs(orig[:,:,1:]-orig[:,:,:-1])
    diag1=np.abs(orig[:,1:,1:]-orig[:,:-1,:-1]);diag2=np.abs(orig[:,1:,:-1]-orig[:,:-1,1:])
    kinds.extend((up,right,diag1,diag2))
    dirs=[(-1,-1),(0,-1),(1,-1),(-1,0)]
    result=[]
    for k,arr in enumerate(kinds):
        ts=[]
        # Compute transition matrices per video-frame then average.
        for dx,dy in dirs:
            mats=[transitions(f,dx,dy,T) for f in arr]
            ts.extend(np.mean(mats,axis=0).tolist())
        result.extend(ts)
    # Per-mode stay-optimal probabilities for 9 intra prediction modes.
    for mode in range(9):
        mask=(ipm==mode)
        if ipm.shape!=ipm2.shape:raise ValueError('ipm and reestimated_ipm must have the same shape')
        result.append(float(np.mean(ipm2[mask]==mode)) if mask.any() else 0.)
    return {f'SUPERB_{i:04d}':float(v) for i,v in enumerate(result)}
