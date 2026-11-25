# Temporal baselines trained on Xi'an and tested once on Beijing (R2P), and the reverse direction (P2R)
import sys,time,json
import numpy as np, pandas as pd
from htnet import *
from xc_io import ITEMS
D=pd.read_pickle('work/feats.pkl'); S=np.load('work/seqs.npz'); ARM,HAND,M=S['ARM'],S['HAND'],S['M']
Y=D[[f'y{i}' for i in ITEMS]].to_numpy()
r=(D.cohort=='REHAB').to_numpy(); p=~r
CONF={'flat':dict(hier=False,ctx=False),'ctx':dict(hier=False,ctx=True),'hier':dict(hier=True,ctx=True)}
SEEDS=[0,1,2]
def run(tr,te,conf,tag):
    st=standardize_fit(ARM[tr],HAND[tr],M[tr]); A,H=apply_std(ARM,HAND,st)
    P1=P2=0
    for s in SEEDS:
        net=train(A[tr],H[tr],Y[tr],M[tr],seed=s,**CONF[conf]); a,b=predict(net,A[te],H[te],M[te]); P1=P1+a/len(SEEDS); P2=P2+b/len(SEEDS)
    np.savez(f'work/dl_{tag}_{conf}.npz',P1=P1,P2=P2,idx=np.flatnonzero(te))
    pred=(P1>0.5).astype(int)+(P2>0.5).astype(int); yt=Y[te]
    e=pred.sum(1)-yt.sum(1); print(tag,conf,'acc',round((pred==yt).mean(),3),'MAE',round(np.abs(e).mean(),2),'bias',round(e.mean(),2),'r',round(np.corrcoef(pred.sum(1),yt.sum(1))[0,1],3),flush=True)
for conf in ['flat','ctx','hier']:
    t=time.time(); run(r,p,conf,'R2P'); print('  time',round(time.time()-t),flush=True)
for conf in ['flat','hier']:
    run(p,r,conf,'P2R')
