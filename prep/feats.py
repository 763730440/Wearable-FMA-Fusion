import sys, numpy as np, pandas as pd, pickle

from xc_io import *
def rng(x): return float(np.percentile(x,95)-np.percentile(x,5)) if len(x)>2 else np.nan
def angle_unwrap(x): return np.degrees(np.unwrap(np.radians(x)))
def feat_sensor(a,s,fs):
    f={}
    if s==1: P=dict(pitch=0,yaw=1,roll=2); acc=slice(3,6); gyr=slice(9,12)
    elif s==2: P=dict(elbow=0,pitch=1,yaw=2,roll=3); acc=slice(4,7); gyr=slice(10,13)
    else: P=dict(pitch=0,wrist=1); 
    for k,c in P.items(): f[f'{k}_rom']=rng(angle_unwrap(a[:,c]) if k in('yaw','roll','pitch') else a[:,c])
    if s in(1,2):
        g=np.linalg.norm(a[:,gyr],axis=1); f['gyr_p95']=float(np.percentile(g,95))
        am=np.linalg.norm(a[:,acc],axis=1); f['acc_dyn_p95']=float(np.percentile(np.abs(am-np.median(am)),95))
    else:
        for k,c in dict(f1=2,f12=3,fall=4).items(): f[f'{k}_rom']=rng(a[:,c])
    return f
# sensor configuration per item from REHAB
CFG={}
for it in ITEMS:
    CFG[it]=[s for s in (1,2,3) if rehab_signal(1,it,s) is not None or rehab_signal(2,it,s) is not None]
def row_feats(get):
    r={}
    for it in ITEMS:
        for s in CFG[it]:
            g=get(it,s)
            if g is None: continue
            a,fs=g
            if len(a)<5: continue
            for k,v in feat_sensor(a,s,fs).items(): r[f'i{it}_s{s}_{k}']=v
    return r
if __name__=='__main__':
    L=np.load(f'{RB}/label.npy')
    import pandas as pd
    xl=pd.read_excel('data/rehab/brs.xlsx',sheet_name='data_set1',header=None).iloc[5:5+222].reset_index(drop=True)
    R=[]
    for i in range(1,223):
        r=row_feats(lambda it,s: rehab_signal(i,it,s)); r['cohort']='REHAB'; r['pid']=f'R{i:03d}'; r['date']=pd.to_datetime(xl.iloc[i-1,55],errors='coerce')
        for j,it in enumerate(ITEMS): r[f'y{it}']=int(L[i-1,j])
        # full 33-item total from xlsx
        r['fma33']=float(sum(xl.iloc[i-1,k+1] for k in range(1,34))); R.append(r)
    for s in pla_sessions():
        if not s['files']: continue
        r=row_feats(lambda it,k: pla_signal(s['files'][(it,k)]) if (it,k) in s['files'] else None)
        r['cohort']='PLA'; r['pid']=s['pid']; r['date']=s['date']
        for it in ITEMS: r[f'y{it}']=s['labels'][it]
        r['fma33']=float(sum(s['labels'].values())); R.append(r)
    D=pd.DataFrame(R); D.to_pickle('work/feats.pkl'); print(D.shape); print(CFG)
