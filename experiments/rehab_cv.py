# patient-grouped 5-fold CV within REHAB (development cohort) for responsiveness and patient-level internal reference
import sys,time; sys.path.insert(0,'work')
import numpy as np, pandas as pd
from htnet import *
from xc_io import ITEMS
from sklearn.model_selection import GroupKFold
D=pd.read_pickle('feats.pkl'); S=np.load('seqs.npz'); ARM,HAND,M=S['ARM'],S['HAND'],S['M']
Y=D[[f'y{i}' for i in ITEMS]].to_numpy(); r=np.flatnonzero((D.cohort=='REHAB').to_numpy())
L=pd.read_csv('rehab_link.csv').sort_values('row'); grp=L.subj.to_numpy()
P1=np.zeros((len(r),27)); P2=np.zeros((len(r),27))
for k,(tr,te) in enumerate(GroupKFold(5).split(r,groups=grp)):
    st=standardize_fit(ARM[r[tr]],HAND[r[tr]],M[r[tr]]); A,H=apply_std(ARM,HAND,st)
    for s in range(3):
        t=time.time(); net=train(A[r[tr]],H[r[tr]],Y[r[tr]],M[r[tr]],seed=s,hier=True,ctx=True); a,b=predict(net,A[r[te]],H[r[te]],M[r[te]])
        P1[te]+=a/3; P2[te]+=b/3; print('fold',k,'seed',s,int(time.time()-t),'s',flush=True)
np.savez('rehab_cv_hier.npz',P1=P1,P2=P2,grp=grp)
# RF grouped CV
exec(open('run.py').read().split('RES={}')[0])
R2=R.copy(); prc=pd.DataFrame(index=R2.index,columns=[f'p{it}' for it in ITEMS])
for tr,te in GroupKFold(5).split(R2,groups=grp):
    prc.iloc[te]=fit_predict(R2.iloc[tr],R2.iloc[te],'rf').to_numpy()
np.save('rehab_cv_rf.npy',prc.astype(int).to_numpy()); print('done',flush=True)
