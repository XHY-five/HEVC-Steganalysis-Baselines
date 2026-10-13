"""Calibrate cover/stego YUV by re-encoding with the user's supplied x265 executable."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description='Recompress reconstructed YUV to calibrated HEVC bitstreams')
    root=Path(__file__).resolve().parents[1]
    ap.add_argument('--encoder',default=str(root/'tools/bin/x265_recompress/x265.exe'))
    ap.add_argument('--yuv-root',required=True,help='Folder with each segment under its own subdirectory')
    ap.add_argument('--out',required=True)
    ap.add_argument('--width',type=int,required=True)
    ap.add_argument('--height',type=int,required=True)
    ap.add_argument('--fps',type=int,required=True)
    ap.add_argument('--qp',type=int,required=True)
    ap.add_argument('--frames',type=int,default=60)
    ap.add_argument('--keyint',type=int,default=12)
    ap.add_argument('--skip-existing',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    encoder=Path(a.encoder).resolve()
    if not encoder.is_file():raise FileNotFoundError(encoder)
    files=sorted(Path(a.yuv_root).resolve().rglob('*.yuv'))
    if not files:raise RuntimeError(f'No .yuv files in {a.yuv_root}')
    if encoder.suffix.lower()=='.exe' and sys.platform!='win32' and not a.dry_run:
        raise RuntimeError('The supplied x265.exe is a Windows binary; run under Windows.')
    outroot=Path(a.out).resolve()
    used=set()
    for raw in files:
        seg=raw.parent.name
        if seg in used:raise ValueError(f'Multiple reconstructed YUVs for {seg}; narrow --yuv-root')
        used.add(seg)
        out=outroot/seg/(seg+'_cal.265')
        if a.skip_existing and out.is_file() and out.stat().st_size>0:continue
        cmd=[str(encoder),'-p','5','--frame-threads','1','--rect',
             '--input',str(raw),'--input-res',f'{a.width}x{a.height}',
             '--fps',str(a.fps),'--output',str(out),'--profile','main',
             '--frames',str(a.frames),'--keyint',str(a.keyint),
             '--no-scenecut','--b-adapt','0','--bframes','0','--qp',str(a.qp)]
        print(' '.join('"'+x+'"' if ' ' in x else x for x in cmd),flush=True)
        if not a.dry_run:
            out.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(cmd,check=True)
            if not out.is_file() or out.stat().st_size==0:raise RuntimeError(f'No calibrated output: {out}')

if __name__=='__main__':main()
