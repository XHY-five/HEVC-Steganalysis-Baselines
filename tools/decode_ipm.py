"""Batch extraction of IPM TXT using the supplied instrumented HM TAppDecoder.exe."""
import argparse
from pathlib import Path
import subprocess
import sys

SUFFIX={'cover':'cover_ipm','cover_cal':'cover_cal_ipm','stego':'steg_ipm','stego_cal':'steg_cal_ipm'}

def iter_bitstreams(root):
    if root.is_file():
        if root.suffix.lower() not in ('.265','.bin','.hevc'):
            raise ValueError(f'Expected a .265/.bin bitstream: {root}')
        yield root.parent.name if root.parent.name != root.stem else root.stem, root
    else:
        grouped={}
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix.lower() in ('.265','.bin','.hevc'):
                segment=p.parent.name if p.parent!=root else p.stem
                if segment in grouped:
                    raise ValueError(f'Multiple bitstreams map to {segment}: {grouped[segment]} and {p}. Use a narrower input root.')
                grouped[segment]=p
        yield from grouped.items()

def main():
    ap=argparse.ArgumentParser(description='Produce original HM IPM TXT files from encoded HEVC bitstreams')
    root=Path(__file__).resolve().parents[1]
    ap.add_argument('--decoder',default=str(root/'tools/bin/hm_ipm/TAppDecoder.exe'))
    ap.add_argument('--bitstreams',required=True,help='Root folder with one .265/.bin per clip')
    ap.add_argument('--out',required=True,help='Decoder TXT directory')
    ap.add_argument('--kind',required=True,choices=tuple(SUFFIX))
    ap.add_argument('--keyint',type=int,default=12)
    ap.add_argument('--recon-yuv-dir',help='Optional: write reconstructed YUV for subsequent calibration re-encoding')
    ap.add_argument('--skip-existing',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    dec=Path(a.decoder).resolve();inputs=Path(a.bitstreams).resolve();target=Path(a.out).resolve()
    if not dec.is_file():raise FileNotFoundError(f'Decoder not found: {dec}')
    if not inputs.exists():raise FileNotFoundError(f'Bitstreams not found: {inputs}')
    rows=list(iter_bitstreams(inputs))
    if not rows:raise RuntimeError(f'No HEVC .265/.bin bitstreams under {inputs}')
    if dec.suffix.lower()=='.exe' and sys.platform!='win32' and not a.dry_run:
        raise RuntimeError('The supplied TAppDecoder.exe is a Windows binary; run this script under Windows.')
    for segment,stream in rows:
        folder=target/segment
        out=folder/f'{segment}_{SUFFIX[a.kind]}.txt'
        if a.skip_existing and out.is_file() and out.stat().st_size>0:
            print('SKIP:',out);continue
        cmd=[str(dec),'-b',str(stream),f'--KeyInt={a.keyint}',f'--OutputIPMFile={out}']
        if a.recon_yuv_dir:
            yuv=Path(a.recon_yuv_dir).resolve()/segment/f'{segment}_{a.kind}.yuv'
            if not a.dry_run:yuv.parent.mkdir(parents=True,exist_ok=True)
            cmd.extend(['-o',str(yuv)])
        print(' '.join('"'+x+'"' if ' ' in x else x for x in cmd),flush=True)
        if a.dry_run:continue
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/f'{segment}_decoder_console.log').open('w',encoding='utf8') as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        if not out.exists() or out.stat().st_size==0:
            raise RuntimeError(f'Decoder returned without the required IPM TXT: {out}')
        print('TXT:',out,flush=True)

if __name__=='__main__':main()
