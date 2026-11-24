"""Controlled missingness experiment on Xi'an (pooled patient-grouped CV test folds).
Recordings of test items are removed under two mechanisms:
  MNAR  - items the clinician scored 0 lose their recording with probability p (others with probability 0.02)
  MCAR  - every item loses its recording with probability p, regardless of score
Methods: RF with median imputation (current practice), RF + zero rule, latent fusion with missing items as flat evidence,
missingness-aware latent fusion with rho estimated by unsupervised EM on the (unlabeled) test fold."""
import pickle, json
from common import *
from exp_a import rf
from irt import fuse, fit_tau
from mirt import MGPCM
from missirt import fuse_m, rho_from_labels, estimate_rho
C = pickle.load(open('s1_rom.pkl', 'rb')); r = RIDX; Y = YALL[r]; R0 = M[r].astype(bool)
# RF prediction for an item whose features are all missing (median-imputed row), per fold
med_pred = {}
for k in range(5):
    tr = C[f'cv{k}']['tr']; Xr = D.iloc[r[tr]]; out = np.zeros(len(ITEMS), int)
    for j, it in enumerate(ITEMS):
        X = Xr[FC[it]].to_numpy(float); y = Y[tr, j]
        if len(np.unique(y)) == 1: out[j] = y[0]; continue
        m = rf().fit(X, y); out[j] = m.predict(np.full((1, X.shape[1]), np.nan))[0]
    med_pred[k] = out
models = {}
for k in range(5):
    sp = C[f'cv{k}']; g = MGPCM(smooth=1.0).fit(Y[sp['tr']]); to, tc = fit_tau(g, sp['oof'], Y[sp['tr']], mask=R0[sp['tr']])
    models[k] = (g, to, tc, rho_from_labels(Y[sp['tr']], R0[sp['tr']]))
print('models ready', flush=True)
res = {}
for mech in ('MNAR', 'MCAR'):
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        acc = {}; mae = {}
        for seed in range(3):
            rng = np.random.default_rng(seed)
            preds = {m: np.zeros_like(Y) for m in ('RF-median', 'RF-zero', 'Fusion-flat', 'Fusion-MA')}
            for k in range(5):
                sp = C[f'cv{k}']; te = sp['te']; Pte = sp['Pte']; yt = Y[te]
                u = rng.random(yt.shape)
                drop = (u < np.where(yt == 0, p, 0.02)) if mech == 'MNAR' else (u < p)
                Rt = R0[te] & ~drop
                base = Pte.argmax(2)
                preds['RF-median'][te] = np.where(Rt, base, med_pred[k][None])
                preds['RF-zero'][te] = np.where(Rt, base, 0)
                g, to, tc, rho0 = models[k]
                preds['Fusion-flat'][te] = fuse(g, Pte, to, tc, mask=Rt)[0].argmax(2)
                rho = estimate_rho(g, Pte, Rt, rho0, to, tc, iters=15)
                preds['Fusion-MA'][te] = fuse_m(g, Pte, Rt, rho, to, tc).argmax(2)
            for m, pr in preds.items():
                acc.setdefault(m, []).append((pr == Y).mean()); mae.setdefault(m, []).append(np.abs(pr.sum(1) - Y.sum(1)).mean())
        res[f'{mech}_{p}'] = {m: dict(acc=round(float(np.mean(acc[m])), 4), mae=round(float(np.mean(mae[m])), 3)) for m in acc}
        print(mech, p, res[f'{mech}_{p}'], flush=True)
json.dump(res, open('res_sim.json', 'w'), indent=1)
