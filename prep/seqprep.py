import sys, numpy as np, pandas as pd
sys.path.insert(0,'work')
from xc_io import *
from feats import CFG
T=64
def unwrap(x): return np.degrees(np.unwrap(np.radians(x)))
def arm_channels(a1,a2):
    # S1: pitch,yaw,roll | acc | gyro ; S2: elbow, pitch,yaw,roll | acc | gyro
    ang1=np.stack([unwrap(a1[:,c]) for c in range(3)],1); ang2=np.stack([a2[:,0]]+[unwrap(a2[:,c]) for c in (1,2,3)],1)
    return ang1,a1[:,3:6],a1[:,9:12],ang2,a2[:,4:7],a2[:,10:13]
def resample(x,n):
    t=np.linspace(0,1,len(x)); u=np.linspace(0,1,n)
    return np.stack([np.interp(u,t,x[:,c]) for c in range(x.shape[1])],1)
def best_bout(env,fs,ang):
    """find movement bouts from activity envelope; return (i0,i1) of bout with largest angular excursion"""
    k=max(1,int(round(0.2*fs))); e=np.convolve(env,np.ones(k)/k,'same')
    base=np.percentile(e,10); thr=base+0.2*(np.percentile(e,98)-base)
    on=e>thr
    if on.sum()<3: return 0,len(env)
    idx=np.flatnonzero(np.diff(np.r_[0,on.astype(int),0]))
    bouts=list(zip(idx[::2],idx[1::2]))
    # merge bouts separated by <0.3 s
    m=[list(bouts[0])]
    for b in bouts[1:]:
        if b[0]-m[-1][1]<0.3*fs: m[-1][1]=b[1]
        else: m.append(list(b))
    pad=int(0.3*fs)
    best=max(m,key=lambda b: np.ptp(ang[max(0,b[0]-pad):b[1]+pad],axis=0).max())
    return max(0,best[0]-pad), min(len(env),best[1]+pad)
def item_seq(get,it,segment):
    s=CFG[it]
    if s==[1,2]:
        g1,g2=get(it,1),get(it,2)
        if g1 is None or g2 is None: return None
        (a1,fs1),(a2,fs2)=g1,g2
        n=min(len(a1),len(a2)); a1,a2=a1[:n],a2[:n]
        if n<5: return None
        ang1,acc1,gy1,ang2,acc2,gy2=arm_channels(a1,a2)
        if segment:
            env=np.linalg.norm(gy1,axis=1)+np.linalg.norm(gy2,axis=1)
            i0,i1=best_bout(env,fs1,np.c_[ang1,ang2]); 
            ang1,acc1,gy1,ang2,acc2,gy2=[z[i0:i1] for z in (ang1,acc1,gy1,ang2,acc2,gy2)]
        ang1=ang1-ang1[0]; ang2=ang2-ang2[0]
        x=np.c_[ang1,acc1,gy1,ang2,acc2,gy2]   # 3+3+3+4+3+3 = 19
    else:
        g=get(it,3)
        if g is None: return None
        a,fs=g
        if len(a)<5: return None
        x=a.copy()
        if segment:
            env=np.abs(np.gradient(a[:,4]))+np.abs(np.gradient(a[:,0]))
            i0,i1=best_bout(env,fs,a[:,[0,1,4]]); x=x[i0:i1]
        x=x-x[0]
    if len(x)<5: return None
    return resample(x,T).astype(np.float32)
if __name__=='__main__':
    seg=True
    D=pd.read_pickle('feats.pkl')
    ARM=np.zeros((len(D),27,T,19),np.float32); HAND=np.zeros((len(D),27,T,5),np.float32); M=np.zeros((len(D),27),bool)
    for r,(c,pid) in enumerate(zip(D.cohort,D.pid)):
        if c=='REHAB':
            i=int(pid[1:]); get=lambda it,s: rehab_signal(i,it,s); sg=False   # REHAB already trimmed by therapist
        else:
            ss=[s for s in pla_sessions() if s['pid']==pid and s['date']==D.date.iloc[r]][0]
            get=lambda it,k,ss=ss: pla_signal(ss['files'][(it,k)]) if (it,k) in ss['files'] else None; sg=True
        for j,it in enumerate(ITEMS):
            x=item_seq(get,it,sg)
            if x is None: continue
            M[r,j]=True
            if x.shape[1]==19: ARM[r,j]=x
            else: HAND[r,j]=x
    np.savez_compressed('seqs.npz',ARM=ARM,HAND=HAND,M=M)
    print(M.mean(), M.sum(0))
