"""Xing et al. (Neurocomputing 2024), TU-only helper layer.

This module does NOT encode HEVC. It consumes TU candidates measured by a
source-level x265/HM hook and produces forced TU decisions. The encoder must
re-run residual, coefficient, reconstruction and entropy processing for the
chosen decision; changing CABAC flags or m_tuDepth alone is insufficient.
"""
from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


FIELDS = ('frame', 'ctu', 'x', 'y', 'cu_size', 'pu_mode', 'pred_mode',
          'tu_bit', 'j_single', 'j_split', 'n4x4')
PLAN_FIELDS = FIELDS + ('target_bit', 'used', 'strategy', 'rho')


@dataclass(frozen=True)
class Candidate:
    frame: int
    ctu: int
    x: int
    y: int
    cu_size: str
    pu_mode: str
    pred_mode: str
    tu_bit: int
    j_single: float | None
    j_split: float | None
    n4x4: int

    @property
    def key(self) -> tuple[int, int, int, int]:
        return self.frame, self.ctu, self.x, self.y

    @property
    def group(self) -> tuple[int, int]:
        return self.frame, self.ctu

    @property
    def distortion(self) -> float:
        if self.j_single is None or self.j_split is None:
            raise ValueError('Embedding needs both j_single and j_split')
        return abs(self.j_single - self.j_split)

    def eligible(self, tb: int) -> bool:
        return (self.cu_size.lower() in ('8', '8x8', '8×8')
                and self.pu_mode.lower() in ('2nx2n', 'size_2nx2n')
                and self.pred_mode.lower() in ('intra', 'i')
                and self.n4x4 > tb)


def read_candidates(path: str | Path, *,
                    needs_cost: bool = False) -> list[Candidate]:
    path = Path(path)
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = set(FIELDS) if needs_cost else set(
            FIELDS) - {'j_single', 'j_split'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f'{path}: CSV requires columns {sorted(required)}')
        result = []
        seen = set()
        for rowno, r in enumerate(reader, 2):
            try:
                v = Candidate(
                    frame=int(r['frame']), ctu=int(r['ctu']),
                    x=int(r['x']), y=int(r['y']), cu_size=r['cu_size'].strip(),
                    pu_mode=r['pu_mode'].strip(), pred_mode=r['pred_mode'].strip(),
                    tu_bit=int(r['tu_bit']),
                    j_single=float(
                        r['j_single']) if r.get(
                        'j_single',
                        '').strip() else None,
                    j_split=float(
                        r['j_split']) if r.get(
                        'j_split',
                        '').strip() else None,
                    n4x4=int(r['n4x4']))
            except (ValueError, TypeError, KeyError) as e:
                raise ValueError(
                    f'{path}:{rowno}: invalid candidate: {e}') from e
            if v.tu_bit not in (0, 1):
                raise ValueError(f'{path}:{rowno}: tu_bit must be 0 or 1')
            if needs_cost and (v.j_single is None or v.j_split is None):
                raise ValueError(f'{path}:{rowno}: missing RD costs')
            if v.j_single is not None and not math.isfinite(v.j_single):
                raise ValueError(f'{path}:{rowno}: non-finite RD cost')
            if v.j_split is not None and not math.isfinite(v.j_split):
                raise ValueError(f'{path}:{rowno}: non-finite RD cost')
            if v.key in seen:
                raise ValueError(
                    f'{path}:{rowno}: duplicate carrier key {v.key}')
            seen.add(v.key)
            result.append(v)
    return result


def groups(candidates: Sequence[Candidate],
           tb: int) -> dict[tuple[int, int], list[Candidate]]:
    out = defaultdict(list)
    n4 = {}
    for c in candidates:
        if not c.eligible(tb):
            continue
        if c.group in n4 and n4[c.group] != c.n4x4:
            raise ValueError(f'inconsistent n4x4 in CTU {c.group}')
        n4[c.group] = c.n4x4
        out[c.group].append(c)
    return {k: sorted(out[k], key=lambda c: (c.y, c.x)) for k in sorted(out)}


def costs(cands: Sequence[Candidate], mode: str) -> list[float]:
    ds = np.array([c.distortion for c in cands], dtype=np.float64)
    if mode == 'raw':
        return np.maximum(ds, 1e-9).tolist()
    low, high = float(np.min(ds)), float(np.max(ds))
    if high == low:
        return np.ones(len(cands)).tolist()
    if mode == 'paper':
        # Eq. (7) is inverted as printed in the paper. Kept intentionally.
        cs = (high - ds) / (high - low)
    elif mode == 'minmax':
        cs = (ds - low) / (high - low)
    else:
        raise ValueError('cost mode must be raw, paper, or minmax')
    return np.maximum(cs, 1e-9).tolist()


def message_capacity(n: int, ratio: float) -> int:
    if not 0 < ratio <= 1:
        raise ValueError('ratio must be in (0,1]')
    return min(n, int(math.floor(n * ratio + 1e-12)))


def _masks(n: int, m: int, height: int,
           seed: int) -> tuple[list[int], dict[int, int]]:
    if not (1 <= m <= n):
        raise ValueError('STC needs 1 <= message length <= number of carriers')
    if not 2 <= height <= 14:
        raise ValueError('constraint height must be 2..14')
    rng = random.Random(seed)
    boundaries = {((j + 1) * n // m) - 1: j for j in range(m)}
    mask = []
    for i in range(n):
        v = rng.randrange(1 << height)
        if i in boundaries:
            v |= 1  # full-rank pivot for each committed syndrome bit
        mask.append(v)
    return mask, boundaries


def _local_seed(seed: int, frame: int, ctu: int) -> int:
    h = hashlib.sha256(
        f'xing-tu-stc-v1|{seed}|{frame}|{ctu}'.encode()).digest()
    return int.from_bytes(h[:8], 'big')


def stc_extract(stego: Sequence[int], message_length: int, *,
                height: int, seed: int) -> list[int]:
    if message_length == 0:
        return []
    masks, boundary = _masks(len(stego), message_length, height, seed)
    state = 0
    bits = []
    for i, b in enumerate(stego):
        if b not in (0, 1):
            raise ValueError('stego must be binary')
        if b:
            state ^= masks[i]
        if i in boundary:
            bits.append(state & 1)
            state >>= 1
    return bits


def stc_embed(cover: Sequence[int], message: Sequence[int], rho: Sequence[float],
              *, height: int = 7, seed: int = 42) -> list[int]:
    n, m = len(cover), len(message)
    if m == 0:
        return list(cover)
    if n != len(rho):
        raise ValueError('rho length mismatch')
    if any(b not in (0, 1) for b in message) or any(b not in (0, 1)
                                                    for b in cover):
        raise ValueError('bits must be 0/1')
    masks, boundary = _masks(n, m, height, seed)
    states = 1 << height
    indices = np.arange(states, dtype=np.int32)
    dp = np.full(states, np.inf, dtype=np.float64)
    dp[0] = 0.0
    # A choice for each incoming trellis state, for traceback.
    choices = np.zeros((n, states), dtype=np.uint8)
    for i, (bit, price, mask) in enumerate(zip(cover, rho, masks)):
        if not math.isfinite(float(price)) or price < 0:
            raise ValueError(f'bad distortion at carrier {i}')
        via_zero = dp + (float(price) if bit == 1 else 0.0)
        via_one = dp[indices ^ mask] + (float(price) if bit == 0 else 0.0)
        use_one = via_one < via_zero
        dp = np.minimum(via_zero, via_one)
        choices[i, :] = use_one
        if i in boundary:
            desired = int(message[boundary[i]])
            # enforce one syndrome bit and right-shift the trellis state
            next_dp = np.full(states, np.inf, dtype=np.float64)
            next_dp[:states // 2] = dp[(np.arange(states // 2) << 1) | desired]
            dp = next_dp
    state = int(np.argmin(dp))
    if not math.isfinite(float(dp[state])):
        raise RuntimeError('STC embedding infeasible')
    result = [0] * n
    for i in range(n - 1, -1, -1):
        if i in boundary:
            state = (state << 1) | int(message[boundary[i]])
        value = int(choices[i, state])
        result[i] = value
        state ^= masks[i] if value else 0
    assert state == 0, 'STC traceback state error'
    if stc_extract(result, m, height=height, seed=seed) != list(message):
        raise AssertionError('STC internal syndrome check failed')
    return result


def write_plan(path: str | Path, records: list[dict]) -> None:
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def to_record(c: Candidate, target: int, used: bool,
              strategy: str, rho: float) -> dict:
    return dict(frame=c.frame, ctu=c.ctu, x=c.x, y=c.y,
                cu_size=c.cu_size, pu_mode=c.pu_mode, pred_mode=c.pred_mode,
                tu_bit=c.tu_bit,
                j_single='' if c.j_single is None else c.j_single,
                j_split='' if c.j_split is None else c.j_split,
                n4x4=c.n4x4,
                target_bit=target, used=int(used), strategy=strategy, rho=f'{rho:.12g}')
