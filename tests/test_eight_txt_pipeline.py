"""End-to-end smoke test: all eight methods ingest the same decoded syntax TXT.

For Zhao, the additional x265 SATD candidate TXT is required as in the uploaded
original code. The synthetic decoded data below are ONLY for exercising code.
"""
from pathlib import Path
import csv,json,subprocess,sys,tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=['cao','shen','zhao','cao-tu','zhai','cao264','dai','dai-pu']


def decoded_lines(label,sid,calibrated=False):
    lines=[]
    for frame in range(3):
        lines.append(f'Frame {frame} begin')
        rectangles=[]
        for y in range(0,32,4):
            for x in range(0,32,4):rectangles.append((x,y,4))
        for y in range(32,64,8):
            for x in range(0,32,8):rectangles.append((x,y,8))
        for y in range(0,32,16):
            for x in range(32,64,16):rectangles.append((x,y,16))
        for y in range(32,64,8):
            for x in range(32,64,8):rectangles.append((x,y,8))
        for i,(x,y,size) in enumerate(rectangles):
            ipm=(i+sid+label*3+frame)%35
            if calibrated and (i+sid)%6==0:ipm=(ipm+4)%35
            depth={4:3,8:3,16:2}[size]
            part=3 if size==4 else 0
            tu_depth=(i+label+frame)%3
            mvx=(i%5)+label; mvy=(i//5%5)+label
            peb=[round((j-8)*.2+label*.03+sid*.01+frame*.01+(i%7)*.1+(0.05 if calibrated else 0),3) for j in range(16)]
            values=','.join(str(v) for v in peb)
            lines.append(f'CTU Address: 0 x={x},y={y} paritionSize={part} depth={depth} size={size}x{size} IPM={ipm} TUdepth={tu_depth} mv_x={mvx} mv_y={mvy} peb=[{values}]')
    return '\n'.join(lines)+'\n'


def candidate_lines(label,sid,calibrated=False):
    lines=[]
    for frame in range(3):
        for i in range(20):
            x=i%5*8; y=i//5*8;ipm=(i+sid+label*2+frame)%35
            costs=[round(10+abs(j-ipm)*.3+label*.01,3) for j in range(35)]
            if calibrated: costs=[v+.03*(j%3) for j,v in enumerate(costs)]
            order=sorted(range(35),key=lambda j:costs[j]);
            lines.append('Frame={f} CTU=0 x={x} y={y} depth=3 partSize=3 size=8x8 coverIPM={m} coverBit=1 MPM=[0,1,2] modeCosts=[{costs}] SATD_sorted=[{order}]'.format(f=frame,x=x,y=y,m=ipm,costs=','.join(map(str,costs)),order=','.join(map(str,order))))
    return '\n'.join(lines)+'\n'


def main():
    with tempfile.TemporaryDirectory(prefix='hevc_txt_pipeline_') as t:
        base=Path(t);all_rows=[];zhao_rows=[]
        for i in range(12):
            label=i%2;sample=f'clip_{i:02d}'
            ori=base/f'{sample}_original.txt';cal=base/f'{sample}_calibrated.txt'
            ori.write_text(decoded_lines(label,i),encoding='utf8')
            cal.write_text(decoded_lines(label,i,True),encoding='utf8')
            origc=base/f'{sample}_candidate_original.txt';calc=base/f'{sample}_candidate_calibrated.txt'
            origc.write_text(candidate_lines(label,i),encoding='utf8')
            calc.write_text(candidate_lines(label,i,True),encoding='utf8')
            row=dict(sample_id=sample,label=label,split='train' if i<8 else 'test',tar='Tar3',payload=0.3,original=str(ori),calibrated=str(cal))
            all_rows.append(row);zhao_rows.append({**row,'original':str(origc),'calibrated':str(calc)})
        def dump(rows,name):
            p=base/name
            with p.open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            return p
        data=dump(all_rows,'syntax_manifest.csv');cand=dump(zhao_rows,'candidate_manifest.csv')
        for method in METHODS:
            cmd=[sys.executable,str(ROOT/method/'run.py'),'--manifest',str(cand if method=='zhao' else data),'--out',str(base/method),'--payload','0.3','--tar','Tar3']
            if method in ('dai','dai-pu'):
                conf=json.loads((ROOT/method/'config.json').read_text())
                conf.update(dict(image_size=32,max_frames=3,base_channels=8,batch_size=2,device='cpu'))
                c=base/f'{method}_test_config.json';c.write_text(json.dumps(conf))
                cmd.extend(['--config',str(c),'--epochs','1'])
            proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
            if proc.returncode:raise AssertionError(f'{method} exited {proc.returncode}:\n{proc.stdout[-4500:]}')
            metrics=json.loads((base/method/'metrics.json').read_text())
            assert metrics['n_test']==4,(method,metrics)
            print(method,'PASS',round(metrics['accuracy'],4))
        # Explicitly check that missing syntax never silently becomes IPM statistics.
        sample=(base/'clip_00_original.txt').read_text()
        legacy='\n'.join(x.split(' TUdepth=')[0] for x in sample.splitlines())+'\n'
        missing=base/'ipm_only.txt';missing.write_text(legacy)
        from_import=ROOT
        sys.path.insert(0,str(from_import))
        from common.decoded import prediction_error_maps,records
        from common.decoded import load_txt
        failed=0
        for call in (lambda:prediction_error_maps({'original':str(missing)}),
                     lambda:records({'original':str(missing)},required=('mv_x',)),
                     lambda:records({'original':str(missing)},required=('tu_depth',))):
            try:call()
            except ValueError:failed+=1
        assert failed==3,f'Expected all three missing-field checks to fail; got {failed}'
        print('Eight method smoke tests and missing-field checks PASS')

if __name__=='__main__':main()
