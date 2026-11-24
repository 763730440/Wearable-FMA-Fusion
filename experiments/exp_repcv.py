"""Repeated patient-grouped 5-fold CV with different random group-to-fold assignments (seeds 1-4; seed 0 is the
GroupKFold split used in the main analysis). Everything is refitted per split: item RFs (with inner grouped OOF),
measurement model, tempering, classical pools, stacking, Transformer and GCN."""
import pickle, json, sys, os, numpy as np
from common import *
from exp_a import stage1, r, grp
from irt import fuse, fit_tau
from mirt import MGPCM
from exp_unc import stage2_proba, tune_pools, lin_pool, log_pool, ds_pool, item_post, conv_indep, latent_parts, latent_total
from learned_fusion import fit_predict
from sklearn.model_selection import GroupKFold

Y = YALL[r]; R0 = M[r].astype(bool)

def folds(seed, k=5):
    ug = np.unique(grp); rng = np.random.default_rng(seed); perm = rng.permutation(ug)
    f = {g: i % k for i, g in enumerate(perm)}; a = np.array([f[g] for g in grp])
    return [(np.flatnonzero(a != i), np.flatnonzero(a == i)) for i in range(k)]

def run(seed):
    fn = f'repcv_{seed}.pkl'
    pred = {}; tot = {}
    for fi, (tr, te) in enumerate(folds(seed)):
        P = stage1(r[tr], r[te], False); oof = np.zeros((len(tr), len(ITEMS), 3))
        for itr, ite in GroupKFold(4).split(tr, groups=grp[tr]): oof[ite] = stage1(r[tr][itr], r[tr][ite], False)
        Rt = R0[te]; out = {}
        Pn = P / P.sum(2, keepdims=True); out['RF'] = (Pn, conv_indep(Pn))
        p = stage2_proba(oof, Y[tr], P); out['Stacking'] = (p, conv_indep(p))
        T, w, tt, ds = tune_pools(oof, Y[tr], R0[tr])
        for n, p in (('Linear opinion pool', lin_pool(P, Rt, T, w)), ('Logarithmic pool', log_pool(P, Rt, T, *tt)), ('Dempster-Shafer', ds_pool(P, Rt, T, *ds))):
            out[n] = (p, conv_indep(p))
        g = MGPCM(smooth=1.0).fit(Y[tr]); to, tc = fit_tau(g, oof, Y[tr], mask=R0[tr])
        item, _ = fuse(g, P, to, tc, mask=Rt); lo, lpost = latent_parts(g, P, to, tc, Rt)
        out['Latent fusion'] = (item, latent_total(g.LP, g.w, lo, lpost))
        for k, nm in (('transformer', 'Transformer'), ('gcn', 'GCN')):
            p = fit_predict(k, oof, R0[tr], Y[tr], grp[tr], P, Rt); out[nm] = (p, conv_indep(p))
        for n, (p, d) in out.items():
            pred.setdefault(n, np.zeros_like(Y))[te] = p.argmax(2); tot.setdefault(n, np.zeros((len(Y), 55)))[te] = d
        print(seed, fi, flush=True)
    pickle.dump(dict(pred=pred, tot=tot), open(fn, 'wb'))

if __name__ == '__main__':
    for s in sys.argv[1:]: run(int(s))
