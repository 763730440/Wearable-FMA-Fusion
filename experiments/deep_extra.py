"""Deep baselines for the fusion paper (Xi'an only): flat TCN (no cross-item context) in pooled CV, and
flat TCN / HT-Net in center transfer. Same training code and settings as the APMR study (3 seeds averaged)."""
import sys, time
import numpy as np, pandas as pd
from htnet import *
from xc_io import ITEMS
from sklearn.model_selection import GroupKFold
D = pd.read_pickle('work/feats.pkl'); S = np.load('work/seqs.npz'); ARM, HAND, M = S['ARM'], S['HAND'], S['M']
Y = D[[f'y{i}' for i in ITEMS]].to_numpy(); r = np.flatnonzero((D.cohort == 'REHAB').to_numpy())
L = pd.read_csv('work/rehab_link.csv').sort_values('row'); grp = L.subj.to_numpy()
crf = pd.read_excel('data/rehab/Rehabilitation_data.xlsx').set_index('SUBJID'); cen = crf.loc[grp, 'CENTER'].to_numpy()
def fitpred(tr, te, hier, ctx):
    st = standardize_fit(ARM[r[tr]], HAND[r[tr]], M[r[tr]]); A, H = apply_std(ARM, HAND, st); P1 = 0; P2 = 0
    for s in range(3):
        t = time.time(); net = train(A[r[tr]], H[r[tr]], Y[r[tr]], M[r[tr]], seed=s, hier=hier, ctx=ctx); a, b = predict(net, A[r[te]], H[r[te]], M[r[te]])
        P1 = P1 + a / 3; P2 = P2 + b / 3; print(' seed', s, int(time.time() - t), 's', flush=True)
    return P1, P2
out = {}
P1 = np.zeros((len(r), 27)); P2 = np.zeros((len(r), 27))
for k, (tr, te) in enumerate(GroupKFold(5).split(r, groups=grp)):
    a, b = fitpred(tr, te, False, False); P1[te] = a; P2[te] = b; print('flat cv fold', k, flush=True)
np.savez('work/deep_cv_flat.npz', P1=P1, P2=P2)
for name, a_c, b_c in (('c1to2', 1, 2), ('c2to1', 2, 1)):
    tr = np.flatnonzero(cen == a_c); te = np.flatnonzero(cen == b_c)
    for kind, h, c in (('flat', False, False), ('hier', True, True)):
        a, b = fitpred(tr, te, h, c); np.savez(f'work/deep_{name}_{kind}.npz', P1=a, P2=b, te=te); print(name, kind, flush=True)
