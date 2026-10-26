"""Tar4 TU-only two-pass x265 4.1 source-patch pipeline.

IMPORTANT: TU bits are read from the FINAL x265-selected encoding tree,
not yet extracted independently by HM decoder CABAC parsing.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .tu_core import read_candidates
from .tu_embed import embed
from .tu_extract import extract
from .tu_verify import verify
from .tu_videos import discover


def need(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Missing/empty file: {path}')


def cmdrun(args, env=None, cwd=None):
    print('>>>', subprocess.list2cmdline([str(a) for a in args]))
    p = subprocess.run([str(a) for a in args], text=True, errors='replace',
                       encoding='utf-8', stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, env=env, cwd=cwd)
    print(p.stdout[-5000:])
    if p.returncode:
        raise RuntimeError(f'Command exit={p.returncode}, see log above')


def encode_command(exe, video, output, qp, frames, keyint, preset, rd):
    return [exe, '-p', preset, '--frame-threads', '1', '--no-wpp',
            '--rect', '--rd', str(rd), '--limit-tu', '0',
            '--input', video.input, '--input-res', f'{video.width}x{video.height}',
            '--fps', str(video.fps), '--output', output, '--profile', 'main',
            '--frames', str(frames), '--keyint', str(keyint), '--no-scenecut',
            '--b-adapt', '0', '--bframes', '0', '--qp', str(qp)]


def filter_candidates(raw: Path, filtered: Path) -> dict:
    """Only finalized CU8, intra 2Nx2N, BOTH finite RD costs may carry TU payload."""
    need(raw)
    rows = []
    with raw.open(encoding='utf-8-sig', newline='') as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            raise ValueError('Bad candidate CSV')
        for r in rdr:
            if not r['j_single'].strip() or not r['j_split'].strip():
                continue
            if r['cu_size'] != '8x8' or r['pu_mode'].lower() != '2nx2n':
                continue
            rows.append(r)
    filtered.parent.mkdir(parents=True, exist_ok=True)
    with filtered.open('w', encoding='utf-8', newline='') as f:
        out = csv.DictWriter(f, fieldnames=rdr.fieldnames)
        out.writeheader()
        out.writerows(sorted(rows, key=lambda r: (
            int(r['frame']), int(r['ctu']), int(r['y']), int(r['x']))))
    parsed = read_candidates(filtered, needs_cost=True)
    return {'raw': sum(1 for _ in raw.open(encoding='utf-8')) - 1,
            'valid_rd': len(parsed)}


def validate_plan_report(path: Path):
    need(path)
    with path.open(encoding='utf-8', newline='') as f:
        records = list(csv.DictReader(f))
    failures = [r for r in records if r['status'] != 'PASS']
    if failures:
        raise RuntimeError(
            f'Encoder did not preserve target TU structures: {len(failures)}/{len(records)} failed; first: {failures[:5]}')
    return {'planned': len(records), 'passed': len(records)}


def pipeline(args):
    root = Path(__file__).resolve().parent.parent
    win = os.name == 'nt'
    x265 = (root / (args.encoder or ('C_Coder/x265_TU.exe' if win else 'C_Coder/x265_TU_linux'))).resolve()
    decoder = (root / args.decoder).resolve()
    inputs = (root / args.input_dir).resolve()
    outs = (root / args.output_dir).resolve()
    phases = [
        'cover',
        'embed',
        'encode',
        'decode',
        'verify'] if args.stage == 'all' else [
        args.stage]
    if 'cover' in phases or 'encode' in phases:
        need(x265)
    videos = discover(inputs, args.frames)
    if args.max_clips:
        videos = videos[:args.max_clips]
    summary = []
    for v in videos:
        base = outs / v.key
        cover_stream = base / 'cover' / f'{v.key}_cover.265'
        cover_raw = base / 'cover' / f'{v.key}_cover_tu_raw.csv'
        cover_candidates = base / 'cover' / f'{v.key}_tu_candidates.csv'
        embed_dir = base / 'embedding'
        plan = embed_dir / 'tu_plan.csv'
        stego_stream = base / 'stego' / f'{v.key}_stego.265'
        stego_final = base / 'stego' / f'{v.key}_final_tu.csv'
        stego_report = base / 'stego' / f'{v.key}_plan_report.csv'
        decoded_ipm = base / 'decoded' / f'{v.key}_decoder_ipm.txt'
        decoded_coeff = base / 'decoded' / f'{v.key}_decoder_coeff.txt'
        print(f'\n==== {v.key}, {v.width}x{v.height}, {v.frames} frames ===')
        stats = {
            'name': v.key,
            'cover': str(cover_stream),
            'stego': str(stego_stream)}
        if 'cover' in phases:
            cover_stream.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            for k in ('X265_TU_PLAN', 'X265_TU_REPORT', 'X265_TU_FINAL'):
                env.pop(k, None)
            env['X265_TU_FINAL'] = str(cover_raw)
            cmdrun(
                encode_command(
                    x265,
                    v,
                    cover_stream,
                    args.qp,
                    v.frames,
                    args.keyint,
                    args.preset,
                    args.rd),
                env,
                root)
            need(cover_stream)
            stats.update(filter_candidates(cover_raw, cover_candidates))
        if 'embed' in phases:
            need(cover_candidates)
            m = embed(cover_candidates, embed_dir, ratio=args.ratio, seed=args.seed,
                      tb=args.tb, tstc=args.tstc, height=args.height, cost_mode=args.cost_mode)
            stats['capacity'] = m['message_length']
            stats['modified'] = m['modifications']
            if m['message_length'] == 0:
                raise RuntimeError(
                    '0 TU payload; try --keyint 1 or different content/QP, do not fake output')
        if 'encode' in phases:
            need(plan)
            stego_stream.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                X265_TU_PLAN=str(plan),
                X265_TU_FINAL=str(stego_final),
                X265_TU_REPORT=str(stego_report))
            cmdrun(
                encode_command(
                    x265,
                    v,
                    stego_stream,
                    args.qp,
                    v.frames,
                    args.keyint,
                    args.preset,
                    args.rd),
                env,
                root)
            need(stego_stream)
            stats.update(validate_plan_report(stego_report))
        if 'decode' in phases:
            need(stego_stream)
            if not decoder.is_file():
                raise FileNotFoundError(
                    'Original HM TAppDecoder.exe not found')
            decoded_ipm.parent.mkdir(parents=True, exist_ok=True)
            cmdrun([decoder, '-b', stego_stream, '--KeyInt=' + str(args.keyint),
                   '--OutputIPMFile=' + str(decoded_ipm),
                    '--OutputCoeffFile=' + str(decoded_coeff)], cwd=root)
            print(
                'HM bitstream decode completed. This is NOT independent TU-bit extraction.')
        if 'verify' in phases:
            need(plan)
            need(stego_final)
            # NOTE: encoded CU tree trace, NOT TU tree from CABAC decoder.
            report = verify(plan, stego_final, args.tb)
            if not report['verified']:
                raise RuntimeError(
                    f'Final chosen x265 TU positions changed: {json.dumps(report,ensure_ascii=False)[:1800]}')
            recovered = base / 'stego' / 'recovered_bits_from_final_encoder_tree.txt'
            ext = extract(stego_final, recovered, ratio=args.ratio, seed=args.seed,
                          tb=args.tb, tstc=args.tstc, height=args.height,
                          manifest=embed_dir / 'manifest.json')
            stats.update(
                encoder_final_tree_verified=True,
                recovered_bits=ext['message_length'])
        base.mkdir(parents=True, exist_ok=True)
        (base / 'summary.json').write_text(json.dumps(stats,
                                                      ensure_ascii=False, indent=2), encoding='utf-8')
        summary.append(stats)
    return summary


def main(default_stage='all'):
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument(
        '--stage',
        choices=[
            'all',
            'cover',
            'embed',
            'encode',
            'decode',
            'verify'],
        default=default_stage)
    a.add_argument('--input-dir', default='test_video')
    a.add_argument('--output-dir', default='TU_results')
    a.add_argument(
        '--encoder',
        help='relative to tar4, default C_Coder/x265_TU.exe on Windows')
    a.add_argument('--decoder', default='Decoder/TAppDecoder.exe')
    a.add_argument('--qp', type=int, default=35)
    a.add_argument('--frames', type=int, default=60)
    a.add_argument('--max-clips', type=int, default=0)
    a.add_argument('--keyint', type=int, default=12)
    a.add_argument('--preset', default='5')
    a.add_argument('--rd', type=int, default=5)
    a.add_argument('--ratio', type=float, default=.3)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--tb', type=int, default=8)
    a.add_argument('--tstc', type=int, default=16)
    a.add_argument('--height', type=int, default=7)
    a.add_argument(
        '--cost-mode',
        choices=[
            'raw',
            'paper',
            'minmax'],
        default='paper')
    v = a.parse_args()
    result = pipeline(v)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
