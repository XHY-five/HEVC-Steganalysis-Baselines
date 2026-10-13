"""Audit eight method inputs, using the same paired decoder TXT manifest.

Zhao is the only method requiring instrumented x265 SATD candidate TXT as
provided with the original uploader's custom encoder.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common.data import read_manifest
from common.decoded import records,pair

REQUIRED={
    'cao': ('depth','partSize','ipm'),
    'shen': ('depth','partSize','ipm'),
    'cao-tu': ('depth','partSize','tu_depth'),
    'zhai': ('mv_x','mv_y','partSize'),
    'dai': ('ipm',),
    'dai-pu': ('depth','partSize','ipm'),
    'cao264': ('ipm',),
    'zhao': ('ipm','order','costs')
}
PEB_KEYS=('peb','prediction_error','residual_block','residual')

def check(row,method):
    req=REQUIRED[method]
    a=records(row,'original',required=req)
    if method in ('cao','shen','cao264','zhao'):
        b=records(row,'calibrated',required=req if method!='zhao' else ('ipm','order','costs'))
    if method in ('cao264','dai'):
        for r in a:
            if not any(k in r for k in PEB_KEYS):
                raise ValueError('Missing true prediction error / PEB syntax field from HM TXT')
    if method=='cao264':
        for r in b:
            if not any(k in r for k in PEB_KEYS):
                raise ValueError('Missing calibrated PEB syntax field from HM TXT')
    return len(a)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--method',default='all',choices=['all',*REQUIRED])
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--out')
    args=parser.parse_args()
    rows=read_manifest(args.manifest).to_dict('records')
    if args.limit:rows=rows[:args.limit]
    report={}
    for method in (REQUIRED if args.method=='all' else (args.method,)):
        errors=[]
        for r in rows:
            try:check(r,method)
            except Exception as ex:errors.append({'sample_id':r['sample_id'],'error':str(ex)})
        report[method]={'total':len(rows),'ready':len(rows)-len(errors),'errors':errors[:5]}
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if args.out:Path(args.out).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    if any(r['errors'] for r in report.values()):sys.exit(2)

if __name__=='__main__':main()
