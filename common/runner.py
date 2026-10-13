"""Unified feature extraction and classification CLI."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from .data import read_manifest
from .classification import evaluate_feature_csv


def execute(extract_fn,config_path=None):
    ap=argparse.ArgumentParser(description='Steganalysis evaluation')
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--config',default=str(config_path))
    ap.add_argument('--out',default='results')
    ap.add_argument('--stage',choices=['all','extract','train'],default='all')
    ap.add_argument('--payload',type=float,default=None)
    ap.add_argument('--tar',default=None)
    args=ap.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    config['seed']=config.get('seed',42)
    if args.payload is not None and 'params_by_payload' in config:
        profile=config['params_by_payload'].get(f'{args.payload:g}',{})
        config.update(profile)
    output=Path(args.out).resolve();output.mkdir(parents=True,exist_ok=True)
    csv=output/'features.csv'
    if args.stage in ('all','extract'):
        manifest=read_manifest(args.manifest,args.payload,args.tar)
        rows=[]
        for i,row in enumerate(manifest.to_dict('records')):
            try:
                import re
                if str(row.get('frame','')).strip() in ('', 'nan') and re.search(r'_frame_\d+', str(row['sample_id'])):
                    raise ValueError('Sample ID specifies a frame but manifest has no frame column. Provide the decoded-log frame index explicitly; the clip-level TXT would otherwise be reused for every frame.')
                values=extract_fn(row,config)
                if isinstance(values,dict): feature=values
                else:feature={f'f{j:04d}':float(v) for j,v in enumerate(values)}
                record={k:row[k] for k in ('sample_id','split','label','tar','payload') if k in row}
                record.update({str(k):float(v) for k,v in feature.items()})
                rows.append(record)
            except (FileNotFoundError,ValueError,KeyError) as exc:
                raise RuntimeError(f'Feature extraction failed for {row["sample_id"]}: {exc}') from exc
        df=pd.DataFrame(rows)
        if df.empty:raise RuntimeError('No extracted features')
        df.to_csv(csv,index=False)
        print(f'Saved {len(df)} samples and {len(df.columns)} columns: {csv}')
    if args.stage in ('all','train'):
        if not csv.exists():raise FileNotFoundError(f'Extract first: {csv}')
        evaluate_feature_csv(csv,config,output)
