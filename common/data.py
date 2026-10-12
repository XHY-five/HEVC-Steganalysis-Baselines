"""Manifest handling and numeric-array utilities."""
from pathlib import Path
import json
import numpy as np
import pandas as pd


def read_manifest(path, payload=None, tar=None):
    path = Path(path).resolve()
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict) and all(isinstance(v, (int, bool)) for v in data.values()):
            raise ValueError('Label-map JSON lists labels only. Use tools/build_manifest.py to map labels to actual feature paths.')
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(path)
    required = {'sample_id', 'label', 'split'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Manifest lacks columns: {sorted(missing)}')
    df = df.copy()
    df['label'] = df['label'].astype(int)
    if not set(df['label']).issubset({0, 1}):
        raise ValueError('label must be 0 (cover) or 1 (stego)')
    df['split'] = df['split'].astype(str).str.lower()
    if payload is not None:
        if 'payload' not in df: raise ValueError('No payload column in manifest')
        df = df[np.isclose(pd.to_numeric(df['payload'], errors='coerce'), float(payload))]
    if tar is not None:
        if 'tar' not in df: raise ValueError('No tar column in manifest')
        df = df[df['tar'].str.lower() == str(tar).lower()]
    base = path.parent
    for col in ['original', 'calibrated', 'data_path', 'mv_path', 'residual_path', 'pu_path', 'error_path']:
        if col in df:
            df[col] = df[col].fillna('').apply(lambda s: str((base / str(s)).resolve()) if str(s).strip() else '')
    if df.empty: raise ValueError('Selected manifest has no samples')
    if df['sample_id'].duplicated().any():
        raise ValueError('Duplicate sample_id found; each example must be unique')
    return df.reset_index(drop=True)


def read_array(path, key=None):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(str(path))
    if path.suffix.lower() == '.npy': return np.load(path, allow_pickle=False)
    if path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as z:
            if key is None: key = z.files[0]
            if key not in z: raise KeyError(f'{key} not found in {path}; keys={z.files}')
            return np.asarray(z[key])
    raise ValueError(f'Expected .npy or .npz array: {path}')


def resolve_array(record, keys):
    """Loads an array from data_path or the first nonempty specified column."""
    for col in keys:
        path = record.get(col, '')
        if isinstance(path, str) and path:
            return read_array(path, key=None)
    raise ValueError(f'No array input in manifest row: {record.get("sample_id")}')
