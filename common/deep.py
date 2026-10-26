"""Training, validation and evaluation of video steganalysis models."""
from pathlib import Path
import copy,json,random
import numpy as np
import pandas as pd
from .data import read_manifest
from .classification import metrics


def run(factory,config_path):
    import argparse
    import torch
    from torch.utils.data import DataLoader,Dataset
    from torch import nn
    from .decoded import pu_maps,prediction_error_maps,selected_frame
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--config',default=str(config_path))
    ap.add_argument('--out',default='results')
    ap.add_argument('--tar',default=None);ap.add_argument('--payload',type=float,default=None)
    ap.add_argument('--epochs',type=int,default=None)
    args=ap.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    seed=int(config.get('seed',42));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    data=read_manifest(args.manifest,payload=args.payload,tar=args.tar)
    device=torch.device('cuda' if torch.cuda.is_available() and config.get('device','auto')!='cpu' else 'cpu')
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    size=int(config.get('image_size',128));max_frames=int(config.get('max_frames',8))
    class VideoSet(Dataset):
        def __init__(self,df): self.rows=df.to_dict('records')
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            row=self.rows[i]
            # Both networks consume the same instrumented HM decoder TXT interface.
            # The methods retain different *actual* syntax representations:
            # PUSN: PU partition size; CESNet: decoded PEB / prediction error.
            if selected_frame(row) is None and '_frame_' in str(row.get('sample_id','')):
                raise ValueError(f'Manifest must provide decoded frame index for {row["sample_id"]}')
            if config.get('input_kind')=='pu_map':
                arr=pu_maps(row)
            elif config.get('input_kind')=='prediction_error':
                arr=prediction_error_maps(row)
            else:
                raise ValueError(f'Unsupported input_kind={config.get("input_kind")}')
            arr=np.asarray(arr,dtype=np.float32)
            if arr.ndim==2:arr=arr[None,None,:,:]
            elif arr.ndim==3:arr=arr[:,None,:,:]
            elif arr.ndim==4:
                if arr.shape[1]>12 and arr.shape[-1]<=4: arr=arr.transpose(0,3,1,2)
            else:raise ValueError(f'Input should be [T,H,W], [T,C,H,W], got {arr.shape}')
            arr=np.nan_to_num(arr[:max_frames],copy=True)
            if arr.shape[0]==0: raise ValueError('Empty video')
            tensor=torch.from_numpy(arr)
            if config.get('input_kind')=='pu_map':
                tensor=tensor/float(config.get('pu_normalizer',8))
            else:
                clip=float(config.get('clip',32))
                tensor=torch.clamp(tensor,-clip,clip)/clip
            tensor=nn.functional.interpolate(tensor,size=(size,size),mode='nearest' if config.get('input_kind')=='pu_map' else 'bilinear',align_corners=False if config.get('input_kind')!='pu_map' else None)
            if len(tensor)<max_frames:
                tensor=torch.cat([tensor,tensor[-1:].repeat(max_frames-len(tensor),1,1,1)],dim=0)
            return tensor,int(row['label']),str(row['sample_id'])
    tr=data[data.split=='train'];te=data[data.split=='test'];val=data[data.split=='val']
    if len(tr)==0 or len(te)==0:raise ValueError('Expected train/test in manifest')
    if set(tr.label)!={0,1}:raise ValueError('Training set needs both classes')
    batch=int(config.get('batch_size',16))
    loaders={k:DataLoader(VideoSet(df),batch_size=batch,shuffle=k=='train',num_workers=0) for k,df in [('train',tr),('val',val),('test',te)] if len(df)>0}
    model=factory(config).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config.get('learning_rate',1e-4)),weight_decay=float(config.get('weight_decay',0)))
    rate_every=int(config.get('decay_every',10));rate_gamma=float(config.get('decay_gamma',0.8))
    schedule=torch.optim.lr_scheduler.StepLR(optimizer,step_size=rate_every,gamma=rate_gamma)
    epochs=int(args.epochs or config.get('epochs',50))
    best=float('inf'); best_state=None;history=[]
    for ep in range(epochs):
        model.train();losses=[]
        for X,y,_ in loaders['train']:
            X=X.to(device);y=y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(X)
            if logits.ndim==3:
                loss=nn.functional.cross_entropy(logits.reshape(-1,2),y[:,None].expand(-1,logits.shape[1]).reshape(-1))
            else:loss=nn.functional.cross_entropy(logits,y)
            loss.backward();optimizer.step();losses.append(loss.item())
        schedule.step()
        val_loss=float(np.mean(losses))
        if 'val' in loaders:
            model.eval();v=[]
            with torch.no_grad():
                for X,y,_ in loaders['val']:
                    logits=model(X.to(device))
                    if logits.ndim==3:logits=logits.mean(1)
                    v.append(nn.functional.cross_entropy(logits,y.to(device)).item())
            val_loss=float(np.mean(v))
        if val_loss<best:
            best=val_loss;best_state=copy.deepcopy(model.state_dict())
        history.append({'epoch':ep+1,'train_loss':float(np.mean(losses)),'val_loss':val_loss,'lr':optimizer.param_groups[0]['lr']})
        print(f'Epoch {ep+1}/{epochs}  train={np.mean(losses):.5f}  val={val_loss:.5f}')
    if best_state is not None:model.load_state_dict(best_state)
    torch.save({'state_dict':model.state_dict(),'config':config},out/'best.pt')
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    model.eval();ys=[];ps=[];scores=[];names=[]
    with torch.no_grad():
        for X,y,ids in loaders['test']:
            logits=model(X.to(device));
            if logits.ndim==3:
                if config.get('aggregation','mean')=='majority':
                    prob=torch.softmax(logits,dim=-1)[...,1]
                    pred=(prob>=0.5).float().mean(dim=1)>=0.5
                    p=pred.long().cpu().numpy()
                    s=prob.mean(dim=1).cpu().numpy()
                else:
                    prob=torch.softmax(logits,dim=-1)[...,1].mean(dim=1)
                    s=prob.cpu().numpy();p=(s>=0.5).astype(int)
            else:
                s=torch.softmax(logits,dim=1)[:,1].cpu().numpy();p=(s>=0.5).astype(int)
            ys.extend(y.numpy());ps.extend(p);scores.extend(s);names.extend(ids)
    res=metrics(ys,ps,scores)
    pd.DataFrame({'sample_id':names,'label':ys,'prediction':ps,'score_stego':scores}).to_csv(out/'predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    print(json.dumps(res,indent=2))
