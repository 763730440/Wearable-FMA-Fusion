"""Fig 5: shortened assessment and selective scoring."""
import json, pickle, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('figs_data.py').read().split('# ---------------- Fig 2')[0])
from common import *
r = RIDX; Y = YALL[r]; T = Y.sum(1)
S = json.load(open('res_short.json')); KS = [3, 5, 7, 9, 12, 15, 18, 21, 27]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
meth = [('Latent fusion, adaptive', 'Latent fusion, adaptive order', AQUA, 'D'), ('Latent fusion, fixed short form', 'Latent fusion, fixed short form', BLUE, 'o'),
        ('Latent fusion, random items', 'Latent fusion, random items', '#a3a29b', 's'), ('Ridge regression, fixed short form', 'Ridge regression on short form', ORANGE, '^')]
for k, lab, c, mk in meth:
    a.plot(KS, [S['cv'][str(q)][k]['mae'] for q in KS], color=c, marker=mk, ms=4.5, lw=2, label=lab)
a.axhline(1.75, color=SUB, lw=1, ls='--'); a.text(3, 1.85, 'RF, all 27 items', ha='left', va='bottom', fontsize=7, color=SUB)
a.set_xlabel('Items recorded'); a.set_ylabel('Total-score MAE (points)'); a.set_xticks(KS); a.set_ylim(0, 6.2)
a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.legend(frameon=False, fontsize=6.8, loc='upper right')
a.set_title('A  Shortened assessment', loc='left', fontsize=8.5, fontweight='bold')
L = {f'cv{k}': pickle.load(open(f'unc_cv{k}.pkl', 'rb')) for k in range(5)}; LL = pickle.load(open('unc_learned.pkl', 'rb')); S1 = pickle.load(open('s1_rom.pkl', 'rb'))
def conv(p):
    d = np.zeros((p.shape[0], 55)); d[:, 0] = 1
    for j in range(27):
        nd = d * p[:, j, :1]; nd[:, 1:] += d[:, :-1] * p[:, j, 1:2]; nd[:, 2:] += d[:, :-2] * p[:, j, 2:3]; d = nd
    return d
curves = [('Latent fusion', 'Latent fusion', AQUA), ('Stacking', 'Stacking', BLUE), ('transformer', 'Transformer', YELLOW), ('gcn', 'GCN', '#8a5cd0'), ('RF', 'RF', '#a3a29b')]
fr = np.linspace(0, 0.5, 26)
for key, lab, c in curves:
    tot = np.zeros((len(Y), 55)); pred = np.zeros(len(Y))
    for k in range(5):
        o = L[f'cv{k}']; te = o['te']
        if key in ('transformer', 'gcn'): p = LL[f'cv{k}']['prob'][key]; d = conv(p); pr = p.argmax(2)
        elif key == 'RF': p = S1[f'cv{k}']['Pte'] / S1[f'cv{k}']['Pte'].sum(2, keepdims=True); d = conv(p); pr = p.argmax(2)
        else: d = o['tot'][key]; pr = o['pred'][key]
        tot[te] = d; pred[te] = pr.sum(1)
    t = np.arange(55); mu = tot @ t; sd = np.sqrt(tot @ t ** 2 - mu ** 2); err = np.abs(pred - T); o = np.argsort(-sd)
    y = []
    for f in fr:
        keep = np.ones(len(Y), bool); keep[o[:int(round(f * len(Y)))]] = False; y.append(err[keep].mean())
    b.plot(fr * 100, y, color=c, lw=2, label=lab)
b.set_xlabel('Assessments referred to the therapist (%)'); b.set_ylabel('MAE of automatically scored assessments'); b.set_ylim(0, 1.9)
b.yaxis.grid(True, color=GRID, lw=0.6); b.set_axisbelow(True); b.legend(frameon=False, fontsize=6.8, loc='upper right', ncol=2)
b.set_title('B  Referral by posterior uncertainty', loc='left', fontsize=8.5, fontweight='bold')
plt.tight_layout(); plt.savefig('fig5_short_triage.png', dpi=300); plt.savefig('fig5_short_triage.pdf')
