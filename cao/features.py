"""Cao 11-dimensional calibration features from the supplied HM IPM TXT logs."""
from collections import defaultdict
from functools import lru_cache
import re
import numpy as np

LOG_RE = re.compile(
    r"CTU Address:\s*(?P<ctu>\d+)\s+x=(?P<x>\d+),y=(?P<y>\d+)\s+paritionSize=(?P<ps>\d+)\s+depth=(?P<depth>\d+)\s+size=(?P<size>[0-9]+x[0-9]+)\s+IPM=(?P<ipm>\d+)"
)

def parse_ipm_log(path):
    """返回 records 列表：每条含 frame, ctu, x,y, partSize, depth, size, ipm"""
    records, frame = [], None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("Frame") and "begin" in s:
                # 允许没有 'Frame x end'，frame 会保持到下一个 begin
                toks = s.split()
                try:
                    frame = int(toks[1])
                except:
                    frame = 0 if frame is None else frame
                continue
            m = LOG_RE.search(s)
            if m and frame is not None:
                g = m.groupdict()
                records.append({
                    "frame": frame,
                    "ctu": int(g["ctu"]),
                    "x": int(g["x"]), "y": int(g["y"]),
                    "partSize": int(g["ps"]),
                    "depth": int(g["depth"]),
                    "size": g["size"],
                    "ipm": int(g["ipm"]),
                })
    return records


def size_to_new_depth(size_str, partSize):
    # 论文中将 CU 尺寸映射到“新深度”：64→0, 32→1, 16→2, 8→(3 或 4[NxN])
    if size_str == "64x64": return 0
    if size_str == "32x32": return 1
    if size_str == "16x16": return 2
    if size_str == "8x8":   return 4 if partSize == 3 else 3
    return 3


def _fill(mat, top, left, h4, w4, val):
    mat[top:top+h4, left:left+w4] = val


def mats_from_records(records, ctu_size=64):
    """按 frame -> {ctu: (depth16x16, ipm16x16)}"""
    by_frame = defaultdict(list)
    for r in records:
        by_frame[r["frame"]].append(r)

    out = {}
    for f in sorted(by_frame.keys()):
        items = by_frame[f]
        by_ctu = defaultdict(list)
        for r in items: by_ctu[r["ctu"]].append(r)
        ctu_mats = {}
        for ctu, lst in by_ctu.items():
            dep = -np.ones((16,16), dtype=np.int16)
            ipm = -np.ones((16,16), dtype=np.int16)
            for r in lst:
                w, h = map(int, r["size"].split("x"))
                w4, h4 = w//4, h//4
                ctu_x = (r["x"] // ctu_size) * ctu_size
                ctu_y = (r["y"] // ctu_size) * ctu_size
                left4 = (r["x"] - ctu_x)//4
                top4  = (r["y"] - ctu_y)//4
                nd = size_to_new_depth(r["size"], r["partSize"])
                _fill(dep, top4, left4, h4, w4, nd)
                _fill(ipm, top4, left4, h4, w4, r["ipm"])
            # 众数填补
            for mat in (dep, ipm):
                miss = (mat < 0)
                if miss.any():
                    vals, cnts = np.unique(mat[~miss], return_counts=True)
                    fill_val = int(vals[np.argmax(cnts)]) if vals.size>0 else 0
                    mat[miss] = fill_val
            ctu_mats[ctu] = (dep, ipm)
        out[f] = ctu_mats
    return out


def compute_11_features(mats_orig, mats_cal, frames_subset=None):
    """严格按论文(11)–(21)：F0–F3 共 11 维"""
    dps_list = []
    C4 = 0; C4p = 0
    Nn   = np.zeros(5, dtype=np.int64)
    Nn_p = np.zeros(5, dtype=np.int64)
    UBn  = np.zeros(5, dtype=np.int64)

    frames = sorted(set(mats_orig.keys()) & set(mats_cal.keys()))
    if frames_subset is not None:
        frames = [f for f in frames if f in frames_subset]

    for f in frames:
        cids = sorted(set(mats_orig[f].keys()) & set(mats_cal[f].keys()))
        for c in cids:
            dep_o, ipm_o = mats_orig[f][c]
            dep_c, ipm_c = mats_cal[f][c]
            # F0: DPS = sum |dep_o - dep_c| over 16x16 (4x4 units)
            dps_list.append(int(np.abs(dep_o - dep_c).sum()))
            # F1: 4→非4、非4→4 的 4x4 单元计数
            C4  += int(((dep_o == 4) & (dep_c != 4)).sum())
            C4p += int(((dep_o != 4) & (dep_c == 4)).sum())
            # F2: 各深度占比（原 vs 校）
            for n in range(5):
                Nn[n]   += int((dep_o == n).sum())
                Nn_p[n] += int((dep_c == n).sum())
            # F3: 不稳定块（深度或IPM变化）
            unstable = (dep_o != dep_c) | (ipm_o != ipm_c)
            for n in range(5):
                UBn[n] += int((unstable & (dep_o == n)).sum())

    total_units = int(Nn.sum()) if Nn.sum() > 0 else 1
    F0_DPSavg = float(np.mean(dps_list)) if dps_list else 0.0
    F1_CP4  = C4  / total_units
    F1_CP4p = C4p / total_units

    F2 = []
    for n in range(1,5):
        DPn   = Nn[n]   / total_units
        DPn_p = Nn_p[n] / total_units
        F2.append(0.0 if DPn == 0 else (DPn - DPn_p) / DPn)

    F3 = []
    for n in range(1,5):
        F3.append((UBn[n] / Nn[n]) if Nn[n] > 0 else 0.0)

    names = ["F0_DPSavg","F1_CP4","F1_CP4p",
             "F2_DDP1","F2_DDP2","F2_DDP3","F2_DDP4",
             "F3_UP1","F3_UP2","F3_UP3","F3_UP4"]
    vals = [F0_DPSavg, F1_CP4, F1_CP4p] + F2 + F3
    return dict(zip(names, vals))


@lru_cache(maxsize=16)
def _mats(path):
    return mats_from_records(parse_ipm_log(path))


def extract(row, config):
    original = _mats(str(row['original']))
    calibrated = _mats(str(row['calibrated']))
    frame = str(row.get('frame', '')).strip()
    subset = None if frame in ('','nan','None') else {int(float(frame))}
    available = set(original) & set(calibrated)
    if subset is not None and not (available & subset):
        raise ValueError(f'No matched frame={frame} in original/calibrated decoder TXT')
    if not available:
        raise ValueError('No common frames in original/calibrated TXT')
    features=compute_11_features(original, calibrated, frames_subset=subset)
    if not all(np.isfinite(v) for v in features.values()):
        raise ValueError('Nonfinite Cao features')
    return features
