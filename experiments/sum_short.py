import pickle, json, numpy as np
from collections import Counter
from common import *
from exp_short import KS
r = RIDX; Y = YALL[r]; T = Y.sum(1)
L = {s: pickle.load(open(f'short_{s}.pkl', 'rb')) for s in [f'cv{k}' for k in range(5)] + ['c1to2', 'c2to1']}

def mets(est, tt, d=None):
    if est.ndim == 2: m = dict(mae=float(np.abs(est - tt[None]).mean(1).mean()), within4=float((np.abs(est - tt[None]) <= 4).mean()))
    else: m = dict(mae=float(np.abs(est - tt).mean()), within4=float((np.abs(est - tt) <= 4).mean()))
    if d is not None:
        c = np.cumsum(d, 1); lo = (c >= .05).argmax(1); hi = (c >= .95).argmax(1)
        m.update(cov90=float(((tt >= lo) & (tt <= hi)).mean()), width90=float((hi - lo).mean()))
    return m

out = {'cv': {}, 'transfer': {}}
names = list(L['cv0']['res'][KS[0]])
for k in KS:
    out['cv'][k] = {}
    for n in names:
        parts = [(L[f'cv{f}']['te'], L[f'cv{f}']['res'][k][n]) for f in range(5)]
        est = np.zeros((5, len(Y))) if parts[0][1][0].ndim == 2 else np.zeros(len(Y)); d = None if parts[0][1][1] is None else np.zeros((len(Y), 55))
        for te, (e, dd) in parts:
            if est.ndim == 2: est[:, te] = e
            else: est[te] = e
            if d is not None: d[te] = dd
        out['cv'][k][n] = mets(est, T, d)
for s in ('c1to2', 'c2to1'):
    o = L[s]; tt = T[o['te']]; out['transfer'][s] = {k: {n: mets(e, tt, d) for n, (e, d) in o['res'][k].items()} for k in KS}
    out['transfer'][s]['form'] = [ITEMS[j] for j in o['form']]
out['forms_cv'] = [[ITEMS[j] for j in L[f'cv{f}']['form'][:12]] for f in range(5)]
first = Counter();
for f in range(5):
    for o in L[f'cv{f}']['orders']: first.update(ITEMS[j] for j in o[:5])
out['adaptive_first5_freq'] = first.most_common(12)
json.dump(out, open('res_short.json', 'w'), indent=1, default=str)
for k in KS:
    print(k, {n.replace('Latent fusion', 'LF').replace(', fixed short form', ' [form]'): (round(v['mae'], 2), round(v['within4'] * 100), round(v.get('cov90', np.nan), 2)) for n, v in out['cv'][k].items()})
for s in out['transfer']:
    for k in KS: print(s, k, {n[:30]: round(v['mae'], 2) for n, v in out['transfer'][s][k].items()})
print('forms', out['forms_cv']); print('adaptive first5', out['adaptive_first5_freq'])
