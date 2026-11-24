"""Few-shot site calibration between the two Xi'an hospitals (Tangdu -> Gaoxin and Gaoxin -> Tangdu).
Source models (item classifiers, 2-D measurement model, tempering) are fitted on the source hospital only.
In the target hospital, m patients are drawn at random (20 draws per m) as hand-scored calibration patients; the rest are
test patients. Calibrated site parameters (L2-penalized towards source values): class-wise evidence offsets beta (2),
tempering exponents (2), and proximal/distal ability-scale offsets (2). Baselines: uncalibrated latent fusion, RF with
beta recalibration, stacking (no site fitting), and stacking re-fitted with the calibration patients added."""
import pickle, json, numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax
import sys, os
def logsumexp(x, axis=None, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0)
    o = m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))
    return o if keepdims else np.squeeze(o, axis=axis)
from common import *
from exp_a import stage2
from irt import fit_tau, item_logp, evidence
from mirt import MGPCM, TH, DIM

C = pickle.load(open('s1_rom.pkl', 'rb')); r = RIDX; Y = YALL[r]; R = M[r].astype(bool)

def run_pair(name):
    sp = C[name]; tr, te_all, P, oof = sp['tr'], sp['te'], sp['Pte'], sp['oof']
    Yt, Rt, gt = Y[te_all], R[te_all], GRP[te_all]; tt = Yt.sum(1)
    fn = f'fs_model_{name}.pkl'
    if os.path.exists(fn): g, to0, tc0 = pickle.load(open(fn, 'rb'))
    else:
        g = MGPCM(smooth=1.0).fit(Y[tr]); to0, tc0 = fit_tau(g, oof, Y[tr], mask=R[tr]); pickle.dump((g, to0, tc0), open(fn, 'wb'))
    LPcache = {}
    def LP_shift(s):
        key = (round(s[0], 4), round(s[1], 4))
        if len(LPcache) > 64: LPcache.clear()
        if key not in LPcache: LPcache[key] = np.stack([item_logp(g.a[j], g.d[j], grid=TH[:, DIM[j]] + s[DIM[j]]) for j in range(len(DIM))])
        return LPcache[key]
    def fuse_site(idx, beta, to, tc, s):
        Pq = np.exp(log_softmax(np.log(P[idx] + 1e-4) + np.r_[0.0, beta][None, None], axis=2))
        LP = LP_shift(s); lo = evidence(Pq, to, Rt[idx]); lc = evidence(Pq, tc, Rt[idx])
        A = logsumexp(LP[None] + lc[:, :, None, :], axis=3); lpost = np.log(g.w + 1e-300)[None] + A.sum(1)
        loo = lpost[:, None, :] - A; loo = loo - logsumexp(loo, axis=2, keepdims=True)
        it = lo + logsumexp(loo[..., None] + LP[None], axis=2)
        return np.exp(it - logsumexp(it, axis=2, keepdims=True))
    def calibrate(idx, which, lam=1.0):
        x0 = np.r_[0.0, 0.0, np.log(to0), np.log(tc0), 0.0, 0.0]; free = {'beta': [0, 1], 'beta+tau': [0, 1, 2, 3], 'full': [0, 1, 2, 3, 4, 5]}[which]
        def unpack(z):
            x = x0.copy(); x[free] = z; return x[:2], np.exp(x[2]), np.exp(x[3]), x[4:]
        def f(z):
            b, to, tc, s = unpack(z); it = fuse_site(idx, b, to, tc, s)
            return -np.log(np.take_along_axis(it, Yt[idx][..., None], 2) + 1e-9).sum() / len(idx) + lam * ((z - x0[free]) ** 2).sum()
        return unpack(minimize(f, x0[free], method='Nelder-Mead', options=dict(maxiter=300, xatol=1e-3, fatol=1e-4)).x)
    def rf_beta(idx, lam=1.0):
        f = lambda b: -(np.take_along_axis(log_softmax(np.log(P[idx] + 1e-4) + np.r_[0.0, b][None, None], axis=2), Yt[idx][..., None], 2)[..., 0]).sum() / len(idx) + lam * (b ** 2).sum()
        return minimize(f, np.zeros(2), method='Nelder-Mead').x
    stack0 = stage2(oof, Y[tr], P)
    ug = np.unique(gt); res = {}
    for m in MS:
        keys = ('Latent fusion', 'cal: beta', 'cal: beta+tau', 'cal: full', 'RF (cal: beta)', 'Stacking', 'Stacking + calibration patients')
        rows = {k: [] for k in keys}
        for d in range(ND if m else 1):
            rng = np.random.default_rng(100 * m + d); calp = rng.choice(ug, m, replace=False) if m else np.array([])
            cal = np.flatnonzero(np.isin(gt, calp)); te = np.flatnonzero(~np.isin(gt, calp)); out = {}
            out['Latent fusion'] = fuse_site(te, np.zeros(2), to0, tc0, np.zeros(2)).argmax(2)
            for w in ('beta', 'beta+tau', 'full'):
                b, to, tc, s = calibrate(cal, w) if m else (np.zeros(2), to0, tc0, np.zeros(2))
                out[f'cal: {w}'] = fuse_site(te, b, to, tc, s).argmax(2)
            b = rf_beta(cal) if m else np.zeros(2)
            out['RF (cal: beta)'] = (np.log(P[te] + 1e-4) + np.r_[0.0, b][None, None]).argmax(2)
            out['Stacking'] = stack0[te]
            # stacking re-fitted with the calibration patients' evidence and labels added (target evidence from source classifiers)
            out['Stacking + calibration patients'] = stage2(np.concatenate([oof, P[cal]]), np.concatenate([Y[tr], Yt[cal]]), P[te]) if m else stack0[te]
            for k, p in out.items():
                er = p.sum(1) - tt[te]; mild = tt[te] >= 40
                rows[k].append((np.abs(er).mean(), (p == Yt[te]).mean(), er.mean(), er[mild].mean() if mild.any() else np.nan))
        res[m] = {k: dict(mae=float(np.mean([x[0] for x in v])), acc=float(np.mean([x[1] for x in v])), bias=float(np.mean([x[2] for x in v])),
                          mild_bias=float(np.nanmean([x[3] for x in v]))) for k, v in rows.items()}
        print(name, m, {k: (round(v['mae'], 2), round(v['acc'] * 100, 1)) for k, v in res[m].items()}, flush=True)
    return res

name = sys.argv[1]; MS = [int(x) for x in sys.argv[2].split(',')]; ND = int(sys.argv[3])
res = run_pair(name)
for m, v in res.items(): json.dump(v, open(f'fs_{name}_{m}.json', 'w'), indent=1)
