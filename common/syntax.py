"""Read machine-readable decoded syntax records or instrumented decoder text logs."""
from pathlib import Path
import json
import re
import csv
import numpy as np

PAIR = re.compile(r'([\w]+)\s*=\s*(\[[^\]]*\]|[^\s,]+)')
FRAME = re.compile(r'^\s*Frame\s+(\d+)\s+begin', re.I)
IPM_RE = re.compile(r'\b(?:coverIPM|IPM)\s*=\s*(-?\d+)')
CTU_RE = re.compile(r'CTU\s+Address:\s*(\d+)')


def _scalar(s):
    s = str(s).strip()
    if s.startswith('[') and s.endswith(']'):
        a = s[1:-1].strip()
        return [float(t) for t in re.split(r'[,;\s]+', a) if t]
    try:
        if re.fullmatch(r'[-+]?\d+', s): return int(s)
        return float(s)
    except ValueError:
        return s


def _clean(d):
    rename = {'Frame':'frame','CTU':'ctu','paritionSize':'partSize','partitionSize':'partSize',
              'coverIPM':'ipm','IPM':'ipm','SATD_sorted':'order','modeCosts':'costs',
              'PU':'pu','TUdepth':'tu_depth','TU_Depth':'tu_depth','size':'size'}
    out = {}
    for k,v in d.items():
        name = rename.get(k,k)
        if v is None or v == '': continue
        out[name] = _scalar(v) if isinstance(v,str) else v
    if 'frame' not in out: out['frame'] = 0
    if 'ctu' not in out: out['ctu'] = 0
    return out


def read_syntax(path):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    if path.suffix.lower() == '.json':
        raw=json.loads(path.read_text(encoding='utf-8'))
        if isinstance(raw,dict): raw=raw.get('blocks',raw.get('records',[]))
        return [_clean(r) for r in raw]
    if path.suffix.lower() == '.jsonl':
        return [_clean(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if path.suffix.lower()=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as f:return [_clean(r) for r in csv.DictReader(f)]
    records=[]; frame=0
    for line in path.read_text(encoding='utf-8',errors='ignore').splitlines():
        m=FRAME.search(line)
        if m: frame=int(m.group(1)); continue
        pairs=dict(PAIR.findall(line))
        if not pairs: continue
        m=CTU_RE.search(line)
        if m:pairs['ctu']=m.group(1)
        if 'frame' not in {k.lower() for k in pairs}:pairs['frame']=str(frame)
        if not (('partSize' in pairs or 'paritionSize' in pairs) or
                ('IPM' in pairs or 'coverIPM' in pairs) or ('mv_x' in pairs) or ('tu_depth' in pairs)):
            continue
        records.append(_clean(pairs))
    if not records:raise ValueError(f'No usable syntax records: {path}')
    return records


def get_size(r):
    size=r.get('size',r.get('cu_size',None))
    if isinstance(size,str) and 'x' in size:
        return int(size.lower().split('x')[0]),int(size.lower().split('x')[1])
    if isinstance(size,(int,float)):return int(size),int(size)
    if 'w' in r and 'h' in r:return int(r['w']),int(r['h'])
    depth=int(r.get('depth',3))
    width=max(4,64//(2**min(depth,4)))
    return width,width


def location(r):
    return (int(r.get('frame',0)),int(r.get('ctu',0)),int(r.get('x',0)),int(r.get('y',0)))


def pair_by_location(before,after):
    b={location(r):r for r in before}
    a={location(r):r for r in after}
    for k in b.keys() & a.keys(): yield b[k],a[k]


def block_grid(records, keys=('ipm','partSize'), block=4):
    """Build per-frame maps from decoded CU records; unknown regions remain zero."""
    if not records:raise ValueError('Empty syntax records')
    width=max(int(r.get('x',0))+get_size(r)[0] for r in records)
    height=max(int(r.get('y',0))+get_size(r)[1] for r in records)
    frames=sorted(set(int(r['frame']) for r in records))
    out=np.zeros((len(frames),len(keys),max(1,int(np.ceil(height/block))),max(1,int(np.ceil(width/block)))),dtype=np.float32)
    indexes={v:i for i,v in enumerate(frames)}
    for r in records:
        x,y=int(r.get('x',0))//block,int(r.get('y',0))//block
        w,h=get_size(r); wi,hi=max(1,w//block),max(1,h//block)
        for c,key in enumerate(keys):
            value=r.get(key,0)
            if isinstance(value,(int,float)):
                out[indexes[int(r['frame'])],c,y:y+hi,x:x+wi]=float(value)
    return out
