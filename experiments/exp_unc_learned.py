"""Transformer / GCN item-token fusion: probabilities (for calibration) and sensor-block dropout, same splits as exp_unc.py."""
import pickle, numpy as np
from common import *
import learned_fusion as LF
from mirt import DIM
C = pickle.load(open('s1_rom.pkl', 'rb')); r = RIDX; Y = YALL[r]; R = M[r].astype(bool)
out = {}
for s in [f'cv{k}' for k in range(5)] + ['c1to2', 'c2to1']:
    sp = C[s]; tr, te = sp['tr'], sp['te']; Xtr = LF.tokens(sp['oof'], R[tr]); o = {'te': te, 'prob': {}, 'block': {}}
    for k in ('transformer', 'gcn'):
        make = (lambda q: LF.ItemTransformer(seed=q)) if k == 'transformer' else (lambda q, A=LF.item_graph(Y[tr]): LF.ItemGCN(A, seed=q))
        ms = [LF._train_one(make, Xtr, Y[tr], GRP[tr], q) for q in (0, 1, 2)]
        def pred(Rt):
            X = LF.tokens(sp['Pte'], Rt); pr = 0
            for m in ms:
                lg = m.forward(X).v; ex = np.exp(lg - lg.max(-1, keepdims=True)); pr = pr + ex / ex.sum(-1, keepdims=True)
            return pr / len(ms)
        o['prob'][k] = pred(R[te])
        for blk, nm in ((0, 'arm'), (1, 'glove')): o['block'].setdefault(nm, {})[k] = pred(R[te] & (DIM != blk)[None]).argmax(2)
    out[s] = o; print(s, flush=True)
pickle.dump(out, open('unc_learned.pkl', 'wb'))
