import numpy as np, glob, os, re
from datetime import datetime
ITEMS=[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,24,25,26,27,28,29,30,31]
RB='data/rehab/Rehab_assessment'
PB=glob.glob('data/beijing/01*')[0]
C1=['pitch1','yaw1','roll1','ax1','ay1','az1','mx1','my1','mz1','wx1','wy1','wz1']
C2=['pitch2-pitch1','pitch2','yaw2','roll2','ax2','ay2','az2','mx2','my2','mz2','wx2','wy2','wz2']
C3=['pitch3','pitch3-pitch1','f1','f12','fAll']
COLS={1:C1,2:C2,3:C3}
def rehab_signal(i,item,s):
    f=f'{RB}/d01_raw_data/{i:03d}/{item:03d}_{s}.dat'
    if not os.path.exists(f) or os.path.getsize(f)==0: return None
    return np.loadtxt(f,ndmin=2), 50.0
def pla_signal(path):
    t=open(path,encoding='utf-8-sig').read().strip().split(';')
    rows=[r.split(',') for r in t[1:] if r.strip()]
    if len(rows)<5: return None
    ts=[r[0] for r in rows]
    def _f(x):
        try: return float(x)
        except ValueError: return np.nan
    a=np.array([[_f(x) for x in r[1:]] for r in rows if len(r)==len(rows[0])])
    if np.isnan(a).any():
        import pandas as _pd
        a=_pd.DataFrame(a).interpolate(limit_direction='both').to_numpy()
        if np.isnan(a).any(): a=np.nan_to_num(a)
    t0=datetime.strptime(ts[0],'%Y/%m/%d %H:%M:%S'); t1=datetime.strptime(ts[-1],'%Y/%m/%d %H:%M:%S')
    # rate: samples per second excluding partial first/last seconds
    from collections import Counter
    c=Counter(ts); full=[v for k,v in c.items() if k not in (ts[0],ts[-1])]
    fs=float(np.median(full)) if full else len(rows)/max(1,(t1-t0).seconds+1)
    return a, fs
def pla_sessions():
    out=[]
    for d in sorted(os.listdir(PB)):
        p=os.path.join(PB,d)
        if not os.path.isdir(p) or d.startswith('linshi'): continue
        for fm in sorted(glob.glob(p+'/FMA_*.csv')):
            stamp=os.path.basename(fm)[4:-4]
            lab={}
            for line in open(fm,encoding='utf-8-sig'):
                m=re.match(r'上肢动作(\d+),(\d+)',line.strip())
                if m: lab[int(m.group(1))]=int(m.group(2))
            files={}
            for it in ITEMS:
                for s in (1,2,3):
                    f=f'{p}/{stamp}-{it}-{s}.txt'
                    if os.path.exists(f): files[(it,s)]=f
            out.append(dict(pid=d,stamp=stamp,date=datetime.strptime(stamp,'%Y-%m-%d %H-%M-%S'),labels=lab,files=files))
    return out
