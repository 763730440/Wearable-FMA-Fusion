"""Summarise the repeated cross-validation (Supplementary Table S2): total-score MAE per method and partition.
Partition 1 is the main GroupKFold split (from unc_cv*.pkl / learned_preds.npz via eval_all.py and exp_unc.py);
partitions 2-5 are repcv_1..4.pkl from exp_repcv.py."""
import json, pickle, numpy as np
from common import *
Y = YALL[RIDX]; t = Y.sum(1)
out = {}
for s in (1, 2, 3, 4):
    z = pickle.load(open(f'repcv_{s}.pkl', 'rb'))
    cov = {}
    for k, d in z['tot'].items():
        c = np.cumsum(d, 1); lo, hi = (c >= 0.05).argmax(1), (c >= 0.95).argmax(1); cov[k] = float(((t >= lo) & (t <= hi)).mean())
    out[f'partition_{s + 1}'] = {k: dict(mae=float(np.abs(p.sum(1) - t).mean()), cov90=cov[k]) for k, p in z['pred'].items()}
json.dump(out, open('res_repcv.json', 'w'), indent=1)
for k in out['partition_2']:
    print(f'{k:22s} MAE', ' '.join(f"{out[f'partition_{s}'][k]['mae']:.3f}" for s in (2, 3, 4, 5)), '| cov90', ' '.join(f"{out[f'partition_{s}'][k]['cov90']:.2f}" for s in (2, 3, 4, 5)))
