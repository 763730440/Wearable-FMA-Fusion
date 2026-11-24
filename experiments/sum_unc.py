import pickle, json, numpy as np
from common import *
from eval_all import boot_diff
r = RIDX; Y = YALL[r]; GR = GRP[r] if len(GRP) != len(Y) else GRP
L = {s: pickle.load(open(f'unc_{s}.pkl', 'rb')) for s in [f'cv{k}' for k in range(5)] + ['c1to2', 'c2to1']}
S1 = pickle.load(open('s1_rom.pkl', 'rb')); LL = pickle.load(open('unc_learned.pkl', 'rb'))
def conv_indep(p):
    d = np.zeros((p.shape[0], 55)); d[:, 0] = 1
    for j in range(27):
        nd = d * p[:, j, :1]; nd[:, 1:] += d[:, :-1] * p[:, j, 1:2]; nd[:, 2:] += d[:, :-2] * p[:, j, 2:3]; d = nd
    return d
for s, o in L.items():
    P = S1[s]['Pte'] / S1[s]['Pte'].sum(2, keepdims=True)
    o['prob']['RF'] = P; o['pred']['RF'] = P.argmax(2); o['tot']['RF'] = conv_indep(P)
    for k, nm in (('transformer', 'Transformer'), ('gcn', 'GCN')):
        p = LL[s]['prob'][k]; o['prob'][nm] = p; o['pred'][nm] = p.argmax(2); o['tot'][nm] = conv_indep(p)
        for b in ('arm', 'glove'): o['block'][b][nm] = LL[s]['block'][b][k]

def ece(p, y, bins=15):
    c = p.max(2).ravel(); ok = (p.argmax(2) == y).ravel(); e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (c > lo) & (c <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(ok[m].mean() - c[m].mean())
    return e

def interval(d, a):
    c = np.cumsum(d, 1); return (c >= a / 2).argmax(1), (c >= 1 - a / 2).argmax(1)

def triage(d, pred, y, fr):
    t = np.arange(d.shape[1]); mu = d @ t; sd = np.sqrt(d @ t ** 2 - mu ** 2)
    err = np.abs(pred.sum(1) - y.sum(1)); o = np.argsort(-sd); k = int(round(fr * len(y)))
    keep = np.ones(len(y), bool); keep[o[:k]] = False
    return float(err[keep].mean())

def summarise(prob, tot, pred, y):
    res = {}
    for n in prob:
        p, d, pr = prob[n], tot[n], pred[n]; e = pr.sum(1) - y.sum(1); tt = y.sum(1)
        m = dict(acc=float((pr == y).mean()), mae=float(np.abs(e).mean()), nll=float(-np.log(np.take_along_axis(p, y[..., None], 2) + 1e-9).mean()),
                 brier=float(((p - np.eye(3)[y]) ** 2).sum(2).mean()), ece=float(ece(p, y)))
        for a in (0.1, 0.05):
            lo, hi = interval(d, a); m[f'cov{int(100 - 100 * a)}'] = float(((tt >= lo) & (tt <= hi)).mean()); m[f'width{int(100 - 100 * a)}'] = float((hi - lo).mean())
        m['triage_mae_auto'] = {str(f): triage(d, pr, y, f) for f in (0.0, 0.1, 0.2, 0.3)}
        c = p.max(2).ravel(); ok = (pr == y).ravel(); th = np.quantile(c, 0.2); m['item_acc_at_80pct'] = float(ok[c >= th].mean())
        res[n] = m
    return res

out = {}
# Xi'an pooled CV
names = list(L['cv0']['prob'])
prob = {n: np.zeros(Y.shape + (3,)) for n in names}; tot = {n: np.zeros((len(Y), 55)) for n in names}; pred = {n: np.zeros_like(Y) for n in L['cv0']['pred']}
blk = {b: {n: np.zeros_like(Y) for n in L['cv0']['block'][b]} for b in ('arm', 'glove')}
for k in range(5):
    o = L[f'cv{k}']; te = o['te']
    for n in names: prob[n][te] = o['prob'][n]; tot[n][te] = o['tot'][n]
    for n in pred: pred[n][te] = o['pred'][n]
    for b in blk:
        for n in blk[b]: blk[b][n][te] = o['block'][b][n]
out['cv'] = summarise(prob, tot, pred, Y)
out['cv_ablation'] = {n: dict(acc=float((p == Y).mean()), mae=float(np.abs(p.sum(1) - Y.sum(1)).mean())) for n, p in pred.items() if n.startswith('Abl')}
g = GRP
out['cv_diff'] = {n: boot_diff(pred['Latent fusion'], pred[n], Y, g) for n in names if n != 'Latent fusion'}   # (other - latent): >0 means latent better on MAE
from mirt import DIM
out['block'] = {}
for b, d in blk.items():
    cols = (DIM == (0 if b == 'arm' else 1))
    out['block'][b] = {n: dict(mae=float(np.abs(p.sum(1) - Y.sum(1)).mean()), acc_dropped=float((p[:, cols] == Y[:, cols]).mean()),
                               acc_all=float((p == Y).mean())) for n, p in d.items()}
# transfer
out['transfer'] = {}
for s in ('c1to2', 'c2to1'):
    o = L[s]; y = Y[o['te']]; out['transfer'][s] = summarise(o['prob'], o['tot'], {n: o['pred'][n] for n in o['prob']}, y)
    out['transfer'][s]['_diff_vs_latent'] = {n: boot_diff(o['pred']['Latent fusion'], o['pred'][n], y, GRP[o['te']]) for n in o['prob'] if n != 'Latent fusion'}
    out['transfer'][s]['_block'] = {b: {n: float(np.abs(p.sum(1) - y.sum(1)).mean()) for n, p in o['block'][b].items()} for b in o['block']}
json.dump(out, open('res_unc.json', 'w'), indent=1)
f = lambda x: f'{x:.3f}'
for n, m in out['cv'].items():
    print(f"{n:38s} acc {m['acc']*100:.1f} mae {m['mae']:.2f} nll {m['nll']:.3f} ece {m['ece']:.3f} cov90 {m['cov90']:.2f} w90 {m['width90']:.1f} cov95 {m['cov95']:.2f} "
          f"triage {[round(v,2) for v in m['triage_mae_auto'].values()]} item@80 {m['item_acc_at_80pct']*100:.1f}")
print(out['cv_ablation'])
for b in out['block']:
    print(b, {n: (round(v['mae'], 2), round(v['acc_dropped'] * 100, 1)) for n, v in out['block'][b].items()})
for s in out['transfer']:
    print(s, {n: (round(v['mae'], 2), round(v['acc'] * 100, 1), round(v['cov90'], 2), round(v['ece'], 3)) for n, v in out['transfer'][s].items() if not n.startswith('_')})
    print('  block', out['transfer'][s]['_block'])
print('diff (other-latent MAE, CI):', {n: [round(x, 2) for x in v['mae']] for n, v in out['cv_diff'].items()})
for s in out['transfer']: print(s, {n: [round(x, 2) for x in v['mae']] for n, v in out['transfer'][s]['_diff_vs_latent'].items()})
