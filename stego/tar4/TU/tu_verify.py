"""Verify forced TU plan against TRUE decoded bitstream TU carrier dump."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from .tu_core import read_candidates, groups


def verify(plan_csv: Path, decoded_csv: Path, tb: int = 8) -> dict:
    with plan_csv.open(newline='', encoding='utf-8-sig') as f:
        plan_reader = csv.DictReader(f)
        if not plan_reader.fieldnames or 'target_bit' not in plan_reader.fieldnames:
            raise ValueError('plan must contain target_bit')
        plan = {(int(row['frame']), int(row['ctu']), int(row['x']), int(row['y'])): int(row['target_bit'])
                for row in plan_reader}
    eligible = groups(read_candidates(decoded_csv), tb)
    actual = {c.key: c.tu_bit for items in eligible.values() for c in items}
    missing = sorted(set(plan) - set(actual))
    extra = sorted(set(actual) - set(plan))
    mismatches = [dict(key=list(k), planned=v, decoded=actual[k])
                  for k, v in sorted(plan.items()) if k in actual and actual[k] != v]
    return dict(verified=(not missing and not extra and not mismatches),
                plan_carriers=len(plan), decoded_carriers=len(actual),
                missing=missing[:50], unexpected=extra[:50], mismatches=mismatches[:50],
                missing_count=len(missing), unexpected_count=len(extra),
                bit_error_count=len(mismatches))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--decoded', type=Path, required=True)
    p.add_argument('--output-json', type=Path)
    p.add_argument('--tb', type=int, default=8)
    a = p.parse_args()
    r = verify(a.plan, a.decoded, a.tb)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if a.output_json:
        a.output_json.write_text(json.dumps(r, indent=2), encoding='utf-8')
    if not r['verified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
