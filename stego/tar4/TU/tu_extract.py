"""Extract TU-only payload from a bitstream decoder's TU candidate CSV dump."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .tu_core import groups, message_capacity, read_candidates, stc_extract, _local_seed


def extract(stego_csv: Path, out_path: Path, *, ratio: float, seed: int,
            tb: int, tstc: int, height: int, manifest: Path | None = None) -> dict:
    cgroups = groups(read_candidates(stego_csv), tb)
    meta = json.loads(
        manifest.read_text(
            encoding='utf-8')) if manifest else None
    known = {(x['frame'], x['ctu']): x for x in meta['groups']} if meta else {}
    if meta:
        keys = set(cgroups)
        expected = {k for k, v in known.items() if v['n']}
        if keys != expected:
            raise ValueError(
                f'carrier CTU mismatch: absent={expected-keys}; unexpected={keys-expected}')
    decoded = []
    rows = []
    for (frame, ctu), cands in cgroups.items():
        n = len(cands)
        possible = message_capacity(n, ratio)
        if meta and known[(frame, ctu)]['n'] != n:
            raise ValueError(f'carrier count changed in (frame,CTU)={(frame,ctu)}; '
                             'blind extraction cannot be guaranteed')
        m = int(known[(frame, ctu)]['m']) if meta else possible
        if m < 0 or m > n:
            raise ValueError('invalid message length in CTU')
        strategy = 'stc' if m > 0 and cands[0].n4x4 > tstc else 'direct'
        if meta and strategy != known[(frame, ctu)]['strategy']:
            raise ValueError(f'strategy mismatch in CTU {(frame,ctu)}')
        bits = [c.tu_bit for c in cands]
        piece = (stc_extract(bits, m, height=height,
                             seed=_local_seed(seed, frame, ctu)) if strategy == 'stc'
                 else bits[:m])
        decoded.extend(piece)
        rows.append(dict(frame=frame, ctu=ctu, n=n, m=m, strategy=strategy))
    msg = ''.join(map(str, decoded))
    if meta:
        if len(msg) != meta['message_length']:
            raise ValueError('extracted length differs from manifest')
        if hashlib.sha256(msg.encode()).hexdigest() != meta['message_sha256']:
            raise ValueError('SHA256 mismatch: decoded payload incorrect')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(msg + '\n', encoding='utf-8')
    return dict(message_length=len(msg), sha256=hashlib.sha256(msg.encode()).hexdigest(),
                matched_manifest=(True if meta else None), groups=rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stego-candidates', type=Path, required=True)
    p.add_argument('--output-bits', type=Path, required=True)
    p.add_argument('--ratio', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tb', type=int, default=8)
    p.add_argument('--tstc', type=int, default=16)
    p.add_argument('--height', type=int, default=7)
    p.add_argument('--manifest', type=Path,
                   help='optional validation metadata; not needed for full-rate blind extraction')
    a = p.parse_args()
    info = extract(a.stego_candidates, a.output_bits, ratio=a.ratio,
                   seed=a.seed, tb=a.tb, tstc=a.tstc, height=a.height,
                   manifest=a.manifest)
    print(json.dumps(
        {k: v for k, v in info.items() if k != 'groups'}, indent=2))


if __name__ == '__main__':
    main()
