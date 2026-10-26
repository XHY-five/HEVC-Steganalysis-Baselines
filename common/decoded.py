"""Utilities shared by method-specific extractors for instrumented HM decoder TXT.

Each extractor keeps its own feature rules; these functions only read and align
records exported from the *same original/calibrated decoder TXT pair*.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from .syntax import read_syntax, get_size, location


@lru_cache(maxsize=64)
def load_txt(path):
    path = Path(path)
    if path.suffix.lower() != '.txt':
        raise ValueError(f'Expected HM decoder TXT, got {path}')
    return tuple(read_syntax(path))


def selected_frame(row):
    value = str(row.get('frame', '')).strip()
    return None if value in ('', 'nan', 'None', 'null') else int(float(value))


def records(row, key='original', required=()):
    path = str(row.get(key, '')).strip()
    if not path or path.lower() == 'nan':
        raise ValueError(f'{key} decoder TXT path is missing')
    items = list(load_txt(path))
    frame = selected_frame(row)
    if frame is not None:
        items = [r for r in items if int(r.get('frame', -1)) == frame]
    if not items:
        raise ValueError(f'No decoded records for {key} frame={frame}: {path}')
    for k in required:
        if not all(k in r for r in items):
            raise ValueError(f'{path} is missing {k} in some decoder records; enable this syntax field in the HM decoder TXT exporter')
    return items


def pair(row, required=()):
    a = records(row, 'original', required)
    b = records(row, 'calibrated', required)
    # Alignment is validated at the caller because calibration may change partitions.
    if not set(int(r['frame']) for r in a) & set(int(r['frame']) for r in b):
        raise ValueError('Original/calibrated decoder TXT have no common frames')
    return a, b


def _frame_geometry(items, grid=4):
    width=max(int(r['x']) + get_size(r)[0] for r in items)
    height=max(int(r['y']) + get_size(r)[1] for r in items)
    return max(1,(height+grid-1)//grid),max(1,(width+grid-1)//grid)


def make_map(items, scalar_func, grid=4, max_cells=10000000):
    """Construct [T,H/grid,W/grid] from true logged syntax; no invented values.

    Returns maps and a coverage mask. An uncovered pixel remains zero, and a
    fully empty map is treated as an error.
    """
    if not items: raise ValueError('Empty decoder records')
    h,w=_frame_geometry(items,grid)
    frames=sorted({int(r['frame']) for r in items})
    if len(frames)*h*w > max_cells:
        raise ValueError(f'Map too large ({len(frames)}*{h}*{w}); narrow your frame range')
    out=np.zeros((len(frames),h,w),dtype=np.float32)
    coverage=np.zeros_like(out,dtype=np.bool_)
    ids={f:i for i,f in enumerate(frames)}
    for r in items:
        f=ids[int(r['frame'])];x=int(r['x'])//grid;y=int(r['y'])//grid
        sw,sh=get_size(r);x2=min(w,x+max(1,(sw+grid-1)//grid));y2=min(h,y+max(1,(sh+grid-1)//grid))
        value=float(scalar_func(r))
        if not np.isfinite(value):raise ValueError('Invalid syntax feature value')
        out[f,y:y2,x:x2]=value
        coverage[f,y:y2,x:x2]=True
    if not coverage.any():raise ValueError('No decoded spatial blocks')
    return out,coverage,frames


def pu_side(r):
    """Return prediction-unit side length in pixels from actual HM partition syntax."""
    if 'partSize' not in r or 'depth' not in r:
        raise ValueError('PU map requires partSize and depth in HM decoder TXT')
    w,h=get_size(r)
    mode=int(r['partSize'])
    if mode==3: return max(4,min(w,h)//2)  # NxN
    return max(4,min(w,h))


def pu_maps(row):
    src=records(row,required=('partSize','depth','ipm'))
    maps,mask,frames=make_map(src,pu_side)
    if not mask.all():
        # Missing boundaries occur in cropped / specially instrumented logs.
        # Force data completeness instead of interpreting missing areas as a valid PU class.
        raise ValueError('PU map contains uncovered image areas; check HM decoder TXT coverage')
    return maps


def _block_residual(r):
    """Use logged PEB data, not IPM differences as a residual surrogate."""
    for key in ('peb','prediction_error','residual_block','residual'):
        if key in r:
            value=r[key]
            if isinstance(value,(int,float)):
                return float(value)
            if isinstance(value,list) and len(value) > 0:
                return float(np.mean(np.asarray(value,dtype=float)))
            raise ValueError(f'Malformed {key} in decoder TXT')
    raise ValueError('Prediction-error data absent from HM TXT (expected peb=[...] or prediction_error=[...]); IPM alone cannot recover the prediction residual')


def prediction_error_maps(row):
    src=records(row,required=('ipm',))
    maps,coverage,frames=make_map(src,_block_residual)
    if not coverage.all():
        raise ValueError('Prediction-error map is incomplete; decoder TXT does not cover the frame')
    return maps


def _residual_abs(r):
    for key in ('peb','residual_block','prediction_error','residual'):
        if key in r:
            v=np.asarray(r[key],dtype=float)
            if not v.size: raise ValueError('Empty PEB data')
            return float(np.mean(np.abs(v)))
    raise ValueError('PEB residual field absent; add peb=[...] output to HM decoder')


def residual_abs_maps(row,key='original'):
    src=records(row,key=key,required=('ipm',))
    return make_map(src,_residual_abs)
