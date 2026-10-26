"""Fit baseline classifiers and write reproducible evaluation outputs."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def metrics(y_true, y_pred, score=None):
    y_true=np.asarray(y_true,dtype=int); y_pred=np.asarray(y_pred,dtype=int)
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    d={'n_test':int(len(y_true)), 'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
       'accuracy':float(accuracy_score(y_true,y_pred)),
       'pacc':float(balanced_accuracy_score(y_true,y_pred)),
       'precision':float(precision_score(y_true,y_pred,zero_division=0)),
       'recall':float(recall_score(y_true,y_pred,zero_division=0)),
       'f1':float(f1_score(y_true,y_pred,zero_division=0)),
       'specificity':float(tn/max(1,tn+fp)),
       'fpr':float(fp/max(1,tn+fp)),
       'fnr':float(fn/max(1,fn+tp))}
    if score is not None and len(np.unique(y_true))==2:
        d['auc']=float(roc_auc_score(y_true,score))
    return d


class SubspaceFisherEnsemble:
    """Random-subspace regularized FLD ensemble with probability voting."""
    def __init__(self, n_estimators=60, subspace_size=None, random_state=42):
        self.n_estimators=n_estimators
        self.subspace_size=subspace_size
        self.random_state=random_state

    def fit(self,X,y):
        from sklearn.covariance import LedoitWolf
        X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=int)
        if len(np.unique(y))!=2:raise ValueError('Training requires cover and stego classes')
        rng=np.random.default_rng(self.random_state)
        dim=X.shape[1]; size=min(dim,self.subspace_size or max(2,int(np.sqrt(dim))))
        self.models=[]
        for i in range(self.n_estimators):
            cols=rng.choice(dim,size=size,replace=False)
            ids=rng.choice(len(y),size=len(y),replace=True)
            if len(np.unique(y[ids]))<2:continue
            sub=X[ids][:,cols]
            center=sub.mean(axis=0); std=sub.std(axis=0);std[std<1e-8]=1
            sub=(sub-center)/std
            model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
            model.fit(sub,y[ids]);self.models.append((cols,center,std,model))
        if not self.models: raise RuntimeError('Could not build any ensemble models')
        return self

    def predict_proba(self,X):
        X=np.asarray(X,dtype=float)
        scores=[m.predict_proba((X[:,c]-mean)/std)[:,1] for c,mean,std,m in self.models]
        p=np.mean(scores,axis=0)
        return np.column_stack([1-p,p])

    def predict(self,X):return (self.predict_proba(X)[:,1]>=0.5).astype(int)


def build_classifier(config):
    if config.get('classifier')=='subspace_fld':
        return SubspaceFisherEnsemble(n_estimators=int(config.get('n_estimators',60)),
                                    subspace_size=config.get('subspace_size'),
                                    random_state=int(config.get('seed',42)))
    opts={'kernel':config.get('kernel','rbf'), 'C':float(config.get('C',1.0)),
          'gamma':config.get('gamma','scale')}
    if opts['kernel']=='poly':
        opts['degree']=int(config.get('degree',3));opts['coef0']=float(config.get('coef0',1.0))
    return make_pipeline(StandardScaler(),SVC(**opts,probability=True,random_state=int(config.get('seed',42))))


def evaluate_feature_csv(csv_path,config,outdir):
    df=pd.read_csv(csv_path)
    if 'split' not in df or 'label' not in df:raise ValueError('Feature table must contain split and label')
    ignored={'sample_id','label','split','tar','payload','group','video','segment','original','calibrated','frame'}
    feats=[x for x in df.columns if x not in ignored and pd.api.types.is_numeric_dtype(df[x])]
    if not feats:raise ValueError('No numeric features to train')
    if df[feats].isnull().any().any():raise ValueError('Feature matrix has NaN; fix invalid samples before evaluation')
    tr=df['split'].isin(['train']);te=df['split'].isin(['test'])
    if not tr.any() or not te.any():raise ValueError('Manifest needs train and test records')
    if set(df.loc[tr,'label'])!={0,1}:raise ValueError('Training set must contain cover and stego')
    Xtr=df.loc[tr,feats].to_numpy(float);Xte=df.loc[te,feats].to_numpy(float)
    ytr=df.loc[tr,'label'].to_numpy(int);yte=df.loc[te,'label'].to_numpy(int)
    model=build_classifier(config);model.fit(Xtr,ytr)
    yp=model.predict(Xte)
    scores=model.predict_proba(Xte)[:,1]
    res=metrics(yte,yp,scores)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'sample_id':df.loc[te,'sample_id'].to_numpy(), 'label':yte,
                  'prediction':yp,'score_stego':scores}).to_csv(outdir/'predictions.csv',index=False)
    joblib.dump({'model':model,'columns':feats,'config':config},outdir/'model.joblib')
    (outdir/'metrics.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    print(json.dumps(res,indent=2));return res
