"""Six RM/RP change-rate features of PU sizes 4, 8 and 16."""
from collections import Counter
from common.syntax import read_syntax,get_size


def counts(records):
    c=Counter()
    for r in records:
        if 'depth' in r:
            depth=int(r['depth']); part=int(r.get('partSize',0))
            if depth==0:size=64
            elif depth==1:size=32
            elif depth==2:size=16
            elif depth==3:size=4 if part==3 else 8
            else:size=get_size(r)[0]
        else:size=get_size(r)[0]
        c[size]+=1
    return c


def extract(row,config):
    b=counts(read_syntax(row['original']))
    a=counts(read_syntax(row['calibrated']))
    nb=sum(b.values());na=sum(a.values())
    if nb==0 or na==0:raise ValueError('No partition records')
    out={}
    eps=float(config.get('eps',1e-10))
    for size in (4,8,16):
        if b[size]==0:
            if config.get('zero_count','zero')=='error':raise ValueError(f'No original PU of size {size}')
            out[f'RM{size}']=0.;out[f'RP{size}']=0.;continue
        out[f'RM{size}']=abs(a[size]-b[size])/max(b[size],eps)
        out[f'RP{size}']=abs((a[size]/na)-(b[size]/nb))/max((b[size]/nb),eps)
    return {k:out[k] for k in ['RM4','RM8','RM16','RP4','RP8','RP16']}
