"""Locked single test on the Beijing cohort of the Xi'an-developed fusion methods. Fitted on all Xi'an data."""
import json, sys
from exp_a import stage1, stage2, r, grp, Y
from common import *
from sklearn.model_selection import GroupKFold
from irt import fuse, fit_tau, adapt_prior
from mirt import MGPCM
pla = PIDX; Yp = YALL[pla]; mk = M[pla].astype(bool); MR = M[r].astype(bool); pid = D.iloc[pla].pid.to_numpy()
out = {}; preds = {}
for tag, geo in (('rom', False), ('geo', True)):
    P = stage1(r, pla, geo); oof = np.zeros((len(r), len(ITEMS), 3))
    for itr, ite in GroupKFold(5).split(r, groups=grp): oof[ite] = stage1(r[itr], r[ite], geo)
    np.savez(f'ext_probs_{tag}.npz', P=P, oof=oof)
    preds[f'A0_{tag}'] = P.argmax(2); preds[f'A2_{tag}'] = stage2(oof, Y, P)
    g = MGPCM(smooth=1.0).fit(Y); to, tc = fit_tau(g, oof, Y, mask=MR)
    preds[f'MIRT_{tag}'] = fuse(g, P, to, tc, mask=mk)[0].argmax(2)
    w = adapt_prior(g, P, tc, mask=mk); preds[f'MIRT+prior_{tag}'] = fuse(g, P, to, tc, w=w, mask=mk)[0].argmax(2)
    print(tag, 'done', flush=True)
t = Yp.sum(1); mild = t >= 40
rng = np.random.default_rng(0); ug = np.unique(pid); idx = {u: np.flatnonzero(pid == u) for u in ug}
ea = np.abs(preds['A0_rom'].sum(1) - t)
for k, p in preds.items():
    e = p.sum(1) - t; eb = np.abs(e); d = []
    for _ in range(2000):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]); d.append(eb[s].mean() - ea[s].mean())
    out[k] = dict(acc=round(float((p == Yp).mean()), 4), mae=round(float(eb.mean()), 3), bias=round(float(e.mean()), 3),
                  w425=round(float((eb <= 4.25).mean()), 3), w725=round(float((eb <= 7.25).mean()), 3),
                  mild_mae=round(float(eb[mild].mean()), 2), sev_mae=round(float(eb[~mild].mean()), 2), sev_bias=round(float(e[~mild].mean()), 2),
                  diff_vs_A0=[round(float((eb - ea).mean()), 3)] + [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])])
    print(k, out[k], flush=True)
json.dump(out, open('res_final_ext.json', 'w'), indent=1); np.savez('pred_final_ext.npz', **preds)
