"""TU-only embedding plan generator, not a substitute for patching x265."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .tu_core import (costs, groups, message_capacity, read_candidates,
                      stc_embed, _local_seed, to_record, write_plan)


def embed(input_csv: Path, outdir: Path, *, ratio: float, seed: int,
          tb: int, tstc: int, height: int, cost_mode: str,
          message: list[int] | None = None) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    cgroups = groups(read_candidates(input_csv, needs_cost=True), tb)
    capacity = sum(message_capacity(len(cs), ratio) for cs in cgroups.values())
    if message is not None and len(message) > capacity:
        raise ValueError(
            f'message length {len(message)} exceeds TU capacity {capacity}')
    rng = random.Random(seed)
    generated = []
    plan = []
    summary = []
    cursor = 0
    modified = 0
    for (frame, ctu), cands in cgroups.items():
        possible = message_capacity(len(cands), ratio)
        remaining = possible if message is None else max(
            0, len(message) - cursor)
        m = min(possible, remaining)
        msg = ([rng.randrange(2) for _ in range(m)] if message is None
               else message[cursor:cursor + m])
        cursor += m
        generated.extend(msg)
        cover = [c.tu_bit for c in cands]
        price = costs(cands, cost_mode)
        # The paper checks CTU 4x4-PU count N (NOT TU-candidate count).
        use_stc = m > 0 and cands[0].n4x4 > tstc
        if use_stc:
            targets = stc_embed(cover, msg, price, height=height,
                                seed=_local_seed(seed, frame, ctu))
            strategy = 'stc'
            used_flags = [True] * len(cands)
        else:
            targets = list(cover)
            targets[:m] = msg
            strategy = 'direct'
            used_flags = [i < m for i in range(len(cands))]
        for c, bit, used, rho in zip(cands, targets, used_flags, price):
            modified += c.tu_bit != bit
            plan.append(to_record(c, bit, used, strategy, rho))
        summary.append(dict(frame=frame, ctu=ctu, n=len(cands), m=m,
                            n4x4=cands[0].n4x4, strategy=strategy))
    if message is not None and cursor != len(message):
        raise AssertionError('not all message bits allocated')
    write_plan(outdir / 'tu_plan.csv', plan)
    message_str = ''.join(map(str, generated))
    (outdir / 'message_bits.txt').write_text(message_str + '\n', encoding='utf-8')
    meta = dict(algorithm='xing-tu-only', format_version=1,
                input=str(input_csv), ratio=ratio, seed=seed, tb=tb,
                tstc=tstc, height=height, cost_mode=cost_mode,
                total_candidates=len(plan), max_capacity=capacity,
                message_length=len(generated), modifications=modified,
                message_sha256=hashlib.sha256(
                    message_str.encode()).hexdigest(),
                groups=summary)
    (outdir / 'manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--ratio', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tb', type=int, default=8)
    p.add_argument('--tstc', type=int, default=16)
    p.add_argument('--height', type=int, default=7)
    p.add_argument(
        '--cost-mode',
        choices=[
            'paper',
            'raw',
            'minmax'],
        default='paper')
    p.add_argument('--message-bits', type=Path,
                   help='ASCII 0/1 file; omit to generate deterministic test bits')
    a = p.parse_args()
    secret = None
    if a.message_bits:
        s = ''.join(a.message_bits.read_text(encoding='utf-8').split())
        if set(s) - {'0', '1'}:
            p.error('--message-bits must contain only 0/1 and whitespace')
        secret = [int(c) for c in s]
    meta = embed(a.candidates, a.outdir, ratio=a.ratio, seed=a.seed,
                 tb=a.tb, tstc=a.tstc, height=a.height,
                 cost_mode=a.cost_mode, message=secret)
    print(json.dumps({k: v for k, v in meta.items() if k != 'groups'},
                     indent=2, ensure_ascii=False))
    print('IMPORTANT: The plan is not an HEVC bitstream. Re-encode via patched x265.')


if __name__ == '__main__':
    main()
