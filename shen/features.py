"""Shen's original RM/RP extraction logic on HM decoded IPM/PU TXT."""
from collections import Counter
from functools import lru_cache
from common.syntax import read_syntax

def counts_to_props(cnt, total):
    """数量→占比字典（无键默认0）"""
    props = {}
    for s in (4,8,16,32,64):
        n = cnt.get(s, 0)
        props[s] = (n / total) if total > 0 else 0.0
    return props


def change_rate(a, b):
    """
    变化率 |b - a| / a；当 a==0 时返回 None（后面会跳过）
    """
    if a == 0:
        return None
    return abs(b - a) / a


@lru_cache(maxsize=16)
def _records(path):
    return read_syntax(path)


def _size_count(records):
    cnt=Counter()
    for r in records:
        if 'partSize' not in r or 'depth' not in r:
            continue
        p,depth=int(r['partSize']),int(r['depth'])
        if depth == 0: cnt[64]+=1
        elif depth == 1: cnt[32]+=1
        elif depth == 2: cnt[16]+=1
        elif depth == 3: cnt[4 if p==3 else 8]+=1
    return cnt,sum(cnt.values())


def extract(row, config):
    original=_records(str(row['original']))
    calibrated=_records(str(row['calibrated']))
    frame=str(row.get('frame','')).strip()
    if frame not in ('','nan','None'):
        target=int(float(frame))
        original=[r for r in original if int(r.get('frame',-1))==target]
        calibrated=[r for r in calibrated if int(r.get('frame',-1))==target]
        if not original or not calibrated:
            raise ValueError(f'No PU records for frame {target}')
    before,total_b=_size_count(original)
    after,total_a=_size_count(calibrated)
    if total_b==0 or total_a==0:
        raise ValueError('No PU records with depth and paritionSize in decoded TXT')
    prop_b=counts_to_props(before,total_b)
    prop_a=counts_to_props(after,total_a)
    features={}
    for size in (4,8,16):
        rm=change_rate(before.get(size,0),after.get(size,0))
        rp=change_rate(prop_b[size],prop_a[size])
        if rm is None or rp is None:
            policy=config.get('zero_count','error')
            if policy=='error':
                raise ValueError(f'Original decoded TXT has no {size}x{size} PUs; missing RM/RP')
            rm=0.0;rp=0.0
        features[f'RM{size}']=float(rm)
        features[f'RP{size}']=float(rp)
    return {k:features[k] for k in ('RM4','RM8','RM16','RP4','RP8','RP16')}
