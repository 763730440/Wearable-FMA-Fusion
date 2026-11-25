# HT-Net, patient-grouped 5-fold CV in the Xi'an development data (main partition) -> work/rehab_cv_hier.npz
import sys,time
import numpy as np, pandas as pd
from htnet import *
from xc_io import ITEMS
from sklearn.model_selection import GroupKFold
D=pd.read_pickle('work/feats.pkl'); S=np.load('work/seqs.npz'); ARM,HAND,M=S['ARM'],S['HAND'],S['M']
Y=D[[f'y{i}' for i in ITEMS]].to_numpy(); r=np.flatnonzero((D.cohort=='REHAB').to_numpy())
L=pd.read_csv('work/rehab_link.csv').sort_values('row'); grp=L.subj.to_numpy()
P1=np.zeros((len(r),27)); P2=np.zeros((len(r),27))
for k,(tr,te) in enumerate(GroupKFold(5).split(r,groups=grp)):
    st=standardize_fit(ARM[r[tr]],HAND[r[tr]],M[r[tr]]); A,H=apply_std(ARM,HAND,st)
    for s in range(3):
        t=time.time(); net=train(A[r[tr]],H[r[tr]],Y[r[tr]],M[r[tr]],seed=s,hier=True,ctx=True); a,b=predict(net,A[r[te]],H[r[te]],M[r[te]])
        P1[te]+=a/3; P2[te]+=b/3; print('fold',k,'seed',s,int(time.time()-t),'s',flush=True)
np.savez('work/rehab_cv_hier.npz',P1=P1,P2=P2,grp=grp)
