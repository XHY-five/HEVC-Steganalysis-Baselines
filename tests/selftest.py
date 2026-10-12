"""End-to-end quick run on temporary synthetic inputs."""
from pathlib import Path
import json,subprocess,sys,csv,tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=('cao','shen','zhao','cao-tu','zhai','cao264','dai','dai-pu')


def sample_records(label,index,calibrated=False):
    records=[]
    for frame in range(3):
        for j in range(16):
            x=(j%4)*16;y=(j//4)*16
            size=['4x4','8x8','16x16'][j%3]
            depth=3 if j%3!=2 else 2
            part=3 if j%3==0 else 0
            ipm=(j+label*3+index)%35
            if calibrated and (j+index)%5==0:
                depth=3;part=0;size='8x8';ipm=(ipm+4)%35
            costs=[10+abs(k-ipm)*2+0.1*index for k in range(35)]
            order=sorted(range(35),key=lambda k:costs[k])
            records.append({'frame':frame,'ctu':0,'x':x,'y':y,'size':size,
                            'depth':depth,'partSize':part,'tu_depth':(j+label)%3,
                            'ipm':ipm,'MPM':[0,1,2], 'order':order,'costs':costs,
                            'mv_x':j%3+label,'mv_y':j//3%3+label})
    return records


def main():
    with tempfile.TemporaryDirectory(prefix='hevc_test_') as temp:
        root=Path(temp);records=[]
        rng=np.random.default_rng(1)
        for i in range(12):
            label=i%2
            sid=f'case_{i:02d}'
            ori=root/f'{sid}.json';cal=root/f'{sid}_cal.json'
            ori.write_text(json.dumps(sample_records(label,i)),encoding='utf-8')
            cal.write_text(json.dumps(sample_records(label,i,True)),encoding='utf-8')
            error=rng.normal(label*.2,1,(3,32,32)).astype('float32')
            datap=root/f'{sid}.npy';np.save(datap,error)
            r=root/f'{sid}_residual.npz'
            residual=error
            reest=residual+rng.normal(0,0.2,(3,32,32)).astype('float32')
            ipm=rng.integers(0,9,size=(3,8,8));ipm2=(ipm+label)%9
            np.savez(r,residual=residual,reestimated_residual=reest,ipm=ipm,reestimated_ipm=ipm2)
            records.append({'sample_id':sid,'label':label,'split':'train' if i<8 else 'test',
                            'tar':'Tar3','payload':'0.3','original':ori,'calibrated':cal,
                            'mv_path':ori,'data_path':datap,'residual_path':r})
        manifest=root/'manifest.csv'
        with manifest.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
        for method in METHODS:
            cmd=[sys.executable,str(ROOT/method/'run.py'),'--manifest',str(manifest),
                 '--payload','0.3','--tar','Tar3','--out',str(root/'result'/method)]
            if method in ('dai','dai-pu'):
                conf=json.loads((ROOT/method/'config.json').read_text())
                conf.update({'image_size':32,'max_frames':3,'base_channels':8,'batch_size':2,'device':'cpu'})
                cfg=root/f'{method}_small.json';cfg.write_text(json.dumps(conf))
                cmd.extend(['--epochs','1','--config',str(cfg)])
            proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
            if proc.returncode:
                print(proc.stdout[-5000:]);raise AssertionError(f'{method} failed')
            m=json.loads((root/'result'/method/'metrics.json').read_text())
            assert m['n_test']==4,(method,m)
            print(method,'PASS',round(m['accuracy'],4))
        print('Eight methods completed successfully.')


if __name__=='__main__':main()
