"""Locked Beijing test of the missingness-aware latent fusion (developed on Xi'an and the Xi'an simulation only)."""
import json
from common import *
from exp_a import Y, r
from irt import fit_tau
from mirt import MGPCM
from missirt import fuse_m, rho_from_labels, estimate_rho
pla = PIDX; Yp = YALL[pla]; Rp = M[pla].astype(bool); MR = M[r].astype(bool); t = Yp.sum(1); sev = t < 40; pid = D.iloc[pla].pid.to_numpy()
rng = np.random.default_rng(0); ug = np.unique(pid); idx = {u: np.flatnonzero(pid == u) for u in ug}
base = np.abs(np.load('pred_final_ext.npz')['A0_rom'].sum(1) - t); out = {}
for tag in ('rom', 'geo'):
    e = np.load(f'ext_probs_{tag}.npz'); P, oof = e['P'], e['oof']
    g = MGPCM(smooth=1.0).fit(Y); to, tc = fit_tau(g, oof, Y, mask=MR)
    rho0 = rho_from_labels(Y, MR); rho = estimate_rho(g, P, Rp, rho0, to, tc)
    p = fuse_m(g, P, Rp, rho, to, tc).argmax(2); er = p.sum(1) - t; eb = np.abs(er); d = []
    for _ in range(2000):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]); d.append(eb[s].mean() - base[s].mean())
    out[f'MIRT-MA_{tag}'] = dict(rho_EM=[round(float(x), 3) for x in rho], acc=round(float((p == Yp).mean()), 4), mae=round(float(eb.mean()), 3),
        bias=round(float(er.mean()), 3), w425=round(float((eb <= 4.25).mean()), 3), w725=round(float((eb <= 7.25).mean()), 3),
        mild_mae=round(float(eb[~sev].mean()), 2), mild_bias=round(float(er[~sev].mean()), 2), sev_mae=round(float(eb[sev].mean()), 2), sev_bias=round(float(er[sev].mean()), 2),
        diff_vs_A0=[round(float((eb - base).mean()), 3)] + [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])])
    print(tag, out[f'MIRT-MA_{tag}'], flush=True)
json.dump(out, open('res_final_ext2.json', 'w'), indent=1)
