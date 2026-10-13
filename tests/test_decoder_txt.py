from pathlib import Path
import tempfile,sys,json
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from common.syntax import read_syntax
import importlib.util

def load(name):
 p=root/name/'features.py'
 spec=importlib.util.spec_from_file_location(name.replace('-','_')+'_module',p)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m.extract

with tempfile.TemporaryDirectory() as tmp:
 tmp=Path(tmp)
 cao_txt=tmp/'cao.txt'
 cal_txt=tmp/'cal.txt'
 # HM frame and CTU records with original typo paritionSize.
 cao_txt.write_text('Frame 0 begin\nCTU Address: 0 x=0,y=0 paritionSize=3 depth=3 size=8x8 IPM=15\nCTU Address: 0 x=8,y=0 paritionSize=0 depth=2 size=16x16 IPM=5\nCTU Address: 0 x=24,y=0 paritionSize=0 depth=3 size=8x8 IPM=4\nFrame 1 begin\nCTU Address: 1 x=0,y=0 paritionSize=3 depth=3 size=8x8 IPM=14\n',encoding='utf8')
 cal_txt.write_text('Frame 0 begin\nCTU Address: 0 x=0,y=0 paritionSize=0 depth=3 size=8x8 IPM=13\nCTU Address: 0 x=8,y=0 paritionSize=0 depth=2 size=16x16 IPM=5\nCTU Address: 0 x=24,y=0 paritionSize=0 depth=3 size=8x8 IPM=4\nFrame 1 begin\nCTU Address: 1 x=0,y=0 paritionSize=3 depth=3 size=8x8 IPM=14\n',encoding='utf8')
 rec=read_syntax(cao_txt)
 assert len(rec)==4,rec
 assert rec[0]['frame']==0 and rec[-1]['frame']==1 and rec[0]['ctu']==0,rec
 for method,expected in [('cao',11),('shen',6)]:
  result=load(method)({'sample_id':'s','original':str(cao_txt),'calibrated':str(cal_txt),'frame':'0'}, {})
  print(method,len(result),result)
  assert len(result)==expected
 candidate=tmp/'candidate.txt';candidate_cal=tmp/'candidate_cal.txt'
 arr=','.join(str(i) for i in range(35));costs=','.join(str(i+1) for i in range(35))
 row=f'Frame=0 CTU=0 x=0 y=0 depth=3 partSize=3 size=8x8 coverIPM=7 coverBit=1 MPM=[0,1,2] modeCosts=[{costs}] SATD_sorted=[{arr}]\n'
 candidate.write_text(row,encoding='utf8');candidate_cal.write_text(row,encoding='utf8')
 rec=read_syntax(candidate)
 assert len(rec)==1 and len(rec[0]['costs'])==35 and len(rec[0]['order'])==35,rec
 out=load('zhao')({'sample_id':'c','original':str(candidate),'calibrated':str(candidate_cal),'frame':'0'}, {'relative_cost_threshold':0.04})
 assert len(out)==39 and abs(sum(out.values())-2)<1e-8
 print('zhao',len(out),'F=',sum(v for k,v in out.items() if k.startswith('F')))
print('ALL TXT TESTS PASS')
