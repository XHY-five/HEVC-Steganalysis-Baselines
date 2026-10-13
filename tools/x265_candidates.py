"""Call the user's instrumented x265 build to emit SATD candidate TXT for Zhao."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    ap=argparse.ArgumentParser(description='Emit Zhao IPM candidate lists using customized x265')
    root=Path(__file__).resolve().parents[1]
    ap.add_argument('--encoder',default=str(root/'tools/bin/x265_candidates/x265.exe'))
    ap.add_argument('--input-yuv',required=True)
    ap.add_argument('--width',type=int,required=True)
    ap.add_argument('--height',type=int,required=True)
    ap.add_argument('--fps',type=int,required=True)
    ap.add_argument('--qp',type=int,required=True)
    ap.add_argument('--frames',type=int,default=60)
    ap.add_argument('--keyint',type=int,default=12)
    ap.add_argument('--input-matchfile',required=True,help='Corresponding HM output *_ipm.txt')
    ap.add_argument('--output-candidate',required=True,help='Output TXT with MPM/modeCosts/SATD_sorted')
    ap.add_argument('--output-bitstream',required=True)
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    paths=(a.encoder,a.input_yuv,a.input_matchfile)
    for p in paths:
        if not Path(p).is_file():raise FileNotFoundError(p)
    if Path(a.encoder).suffix=='.exe' and sys.platform!='win32' and not a.dry_run:
        raise RuntimeError('The supplied customized x265.exe is a Windows binary; run under Windows.')
    for p in (a.output_candidate,a.output_bitstream):
        if not a.dry_run:Path(p).parent.mkdir(parents=True,exist_ok=True)
    cmd=[str(Path(a.encoder).resolve()),'-p','5','--frame-threads','1',
         '--input',str(Path(a.input_yuv).resolve()),'--input-res',f'{a.width}x{a.height}',
         '--fps',str(a.fps),'--output',str(Path(a.output_bitstream).resolve()),
         '--profile','main','--frames',str(a.frames),'--keyint',str(a.keyint),
         '--no-scenecut','--b-adapt','0','--bframes','0','--qp',str(a.qp),
         '--input-matchfile',str(Path(a.input_matchfile).resolve()),
         '--output-candidate',str(Path(a.output_candidate).resolve())]
    print(' '.join('"'+x+'"' if ' ' in x else x for x in cmd))
    if not a.dry_run:subprocess.run(cmd,check=True)

if __name__=='__main__':main()
