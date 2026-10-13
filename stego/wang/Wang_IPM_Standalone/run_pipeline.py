#!/usr/bin/env python3
"""Independent Wang2015 IPM embedding + Tar5-compatible custom x265/HM pipeline.

Python 3.9+, stdlib only. Requires external cover-IPM files and source YUV segments.
Customized EXEs are provisioned into bin/ by tools/setup_tools.ps1.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
CORE = PROJECT / 'algorithm' / 'wang_ipm.py'
BATCH = PROJECT / 'algorithm' / 'batch_wang_ipm.py'


def run(cmd, label):
    print('\n[RUN]', label, '\n', subprocess.list2cmdline([str(x) for x in cmd]), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def get_videos(root):
    """Original Tar5 encoder naming: name_WIDTHxHEIGHT_FPS_...segmentNNN.yuv."""
    found = {}
    for p in root.rglob('*.yuv'):
        stem = p.stem
        parts = stem.split('_')
        if len(parts) < 4:
            print('[WARN] Skip YUV with unrecognized name:', p)
            continue
        try:
            width, height = [int(x) for x in parts[1].lower().split('x')]
            fps = int(parts[2])
            seg_index = re.search(r'(\d{3})$', parts[-1])
            if not seg_index:
                raise ValueError('missing segment suffix')
            segname = parts[0] + 'segment' + seg_index.group(1)
            if segname in found:
                raise ValueError(f'duplicate segment name {segname}')
            found[segname] = (p, width, height, fps)
        except (ValueError, IndexError) as e:
            print('[WARN] Skip YUV:', p, '|', e)
    return found


def check_pe(path):
    if not path.is_file() or path.stat().st_size < 100_000:
        raise FileNotFoundError(f'Customized Windows executable missing: {path}\n'
                                'Run PowerShell: .\\tools\\setup_tools.ps1')
    with path.open('rb') as f:
        if f.read(2) != b'MZ':
            raise ValueError(f'Not a Windows PE executable: {path}')


def segments(root, namefilter):
    if not root.exists():
        raise FileNotFoundError('Missing cover IPM directory: ' + str(root))
    result = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or (namefilter and p.name != namefilter):
            continue
        f = p / (p.name + '_cover_ipm.txt')
        if not f.is_file():
            f = p / (p.name + '_cover_IPM.txt')
        if f.is_file():
            result.append((p.name, f))
    if not result:
        raise FileNotFoundError('No <segment>_cover_ipm.txt found in subdirectories of ' + str(root))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--steps', nargs='+', choices=['embed', 'encode', 'decode', 'verify'],
                    default=['embed', 'encode', 'decode', 'verify'])
    ap.add_argument('--cover-root', type=Path, default=PROJECT/'input'/'cover_ipm')
    ap.add_argument('--video-root', type=Path, default=PROJECT/'input'/'videos')
    ap.add_argument('--output-root', type=Path, default=PROJECT/'outputs')
    ap.add_argument('--ddg-root', type=Path, help='Optional precomputed per-frame Tar5 DDG folders')
    ap.add_argument('--payloads', type=float, nargs='+', default=[0.1, 0.3, 0.5])
    ap.add_argument('--qp', type=int, default=35)
    ap.add_argument('--frames', type=int, default=60)
    ap.add_argument('--keyint', type=int, default=4)
    ap.add_argument('--first-i-frame', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--only-segment', help='e.g. BQMallsegment000')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--strict-rd', action='store_true')
    args = ap.parse_args()

    for ratio in args.payloads:
        if not 0 <= ratio <= 1:
            ap.error('All payload values must be in [0,1]')
    if args.frames <= 0 or args.keyint <= 0:
        ap.error('--frames and --keyint must be positive')

    segs = segments(args.cover_root, args.only_segment)
    video_map = get_videos(args.video_root) if 'encode' in args.steps else {}
    x265 = PROJECT / 'bin' / 'C_Coder' / 'x265.exe'
    decoder = PROJECT / 'bin' / 'Decoder' / 'TAppDecoder.exe'
    if 'encode' in args.steps:
        check_pe(x265)
    if 'decode' in args.steps:
        check_pe(decoder)
    print('[INFO] Segments:', len(segs), 'payloads:', args.payloads, 'steps:', args.steps)
    stats = {'encoded':0, 'decoded':0, 'verified':0, 'skipped':0, 'errors': []}

    if 'embed' in args.steps:
        for ratio in args.payloads:
            for name, cover in segs:
                out = args.output_root/'stego_ipm'/f'payload_{ratio}'/name/f'{name}_stego_IPM.txt'
                if out.exists() and out.stat().st_size and not args.overwrite:
                    print('[SKIP] Embed already done:', out); stats['skipped'] += 1; continue
                cmd = [sys.executable, CORE, 'embed', '--cover', cover, '--output', out,
                       '--payload-ratio', str(ratio), '--intra-period', str(args.keyint),
                       '--first-i-frame', str(args.first_i_frame), '--seed', str(args.seed)]
                if args.ddg_root:
                    ddg = args.ddg_root/name/name
                    if ddg.is_dir(): cmd += ['--ddg-dir', ddg]
                    else: print('[WARN] DDG unavailable for', name, '(using nearest-IPM fallback)')
                if args.strict_rd: cmd += ['--strict-rd']
                run(cmd, f'Wang embed: {name} payload={ratio}')

    for ratio in args.payloads:
        for name, _ in segs:
            stem = name+'_stego_IPM.txt'
            stego = args.output_root/'stego_ipm'/f'payload_{ratio}'/name/stem
            stream = args.output_root/'bitstreams'/f'payload_{ratio}'/name/f'{name}_stego.265'
            ipm = args.output_root/'decoded_ipm'/f'payload_{ratio}'/name/f'{name}_steg_ipm.txt'
            coeff = args.output_root/'decoded_ipm'/f'payload_{ratio}'/name/f'{name}_steg_coeff.txt'
            metadata = Path(str(stego)+'.json')
            recovered = args.output_root/'verified_bits'/f'payload_{ratio}'/name/f'{name}_recovered.bits'

            if 'encode' in args.steps:
                if not stego.is_file(): raise FileNotFoundError(f'Stego IPM missing: {stego}')
                if name not in video_map:
                    raise FileNotFoundError(f'Missing YUV for {name} under {args.video_root}')
                input_yuv, width, height, fps = video_map[name]
                stream.parent.mkdir(parents=True, exist_ok=True)
                if stream.exists() and stream.stat().st_size > 0 and not args.overwrite:
                    print('[SKIP] Encoded stream exists:', stream);stats['skipped']+=1
                else:
                    cmd = [x265, '-p', '5', '--frame-threads','1','--rect',
                           '--input',input_yuv,'--input-res',f'{width}x{height}', '--fps', str(fps),
                           '--output',stream,'--profile','main','--frames',str(args.frames),
                           '--keyint',str(args.keyint),'--no-scenecut','--b-adapt','0',
                           '--bframes','0','--qp',str(args.qp), f'--InputStego={stego.resolve()}']
                    run(cmd, f'Custom x265: {name}, payload={ratio}')
                    if not stream.is_file() or stream.stat().st_size == 0:
                        raise RuntimeError(f'No encoded stream produced: {stream}')
                    stats['encoded']+=1

            if 'decode' in args.steps:
                if not stream.is_file(): raise FileNotFoundError(f'Encoded stream missing: {stream}')
                ipm.parent.mkdir(parents=True, exist_ok=True)
                if ipm.exists() and ipm.stat().st_size > 0 and not args.overwrite:
                    print('[SKIP] Decoded IPM exists:', ipm);stats['skipped']+=1
                else:
                    run([decoder,'-b',stream,f'--KeyInt={args.keyint}',
                         f'--OutputIPMFile={ipm.resolve()}',f'--OutputCoeffFile={coeff.resolve()}'],
                        f'HM decode: {name}, payload={ratio}')
                    if not ipm.is_file() or ipm.stat().st_size == 0:
                        raise RuntimeError(f'No decoder IPM produced: {ipm}')
                    stats['decoded']+=1

            if 'verify' in args.steps:
                if not ipm.is_file(): raise FileNotFoundError(f'Decoder IPM missing: {ipm}')
                if not metadata.is_file(): raise FileNotFoundError(f'Embed metadata missing: {metadata}')
                recovered.parent.mkdir(parents=True, exist_ok=True)
                try:
                    run([sys.executable, CORE, 'extract', '--input',ipm,'--metadata',metadata,
                         '--output-bits',recovered],f'Extract/validate: {name}, payload={ratio}')
                    stats['verified']+=1
                except subprocess.CalledProcessError as exc:
                    stats['errors'].append({'segment':name,'payload':ratio,'error':str(exc)})
                    print('[FAIL] Real-bitstream extraction:',name,ratio,'(check encoder modifications and PU consistency)')
    print('\nFINAL SUMMARY:', json.dumps(stats,indent=2,ensure_ascii=False))
    (args.output_root/'pipeline_summary.json').parent.mkdir(parents=True,exist_ok=True)
    (args.output_root/'pipeline_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')
    if stats['errors']: sys.exit(2)

if __name__ == '__main__':
    main()
