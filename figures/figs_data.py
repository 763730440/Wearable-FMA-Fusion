"""Data figures for the fusion manuscript (Fig 2-4)."""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'; INK = '#0b0b0b'; SUB = '#52514e'; GRID = '#e9e8e4'
DEV2STD = {3: 4, 4: 3, 23: 26, 24: 23, 25: 24, 26: 25}

# ---------------- Fig 2: item hierarchy ---------------------------------------------------------------------------
H = json.load(open('hier_cv.json')); items = [DEV2STD.get(i, i) for i in H['items']]
order = np.argsort(items); it_sorted = [items[i] for i in order]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw=dict(width_ratios=[1.8, 1]))
x = np.arange(len(order)); w = 0.4
a.bar(x - w / 2, np.array(H['prior_only']['per_item'])[order] * 100, w, color='#c3c2b7', label=f"Most frequent score ({H['prior_only']['acc']*100:.0f}%)")
a.bar(x + w / 2, np.array(H['2D']['per_item'])[order] * 100, w, color=AQUA, label=f"Predicted from the other 26 items ({H['2D']['acc']*100:.0f}%)")
a.set_xticks(x); a.set_xticklabels(it_sorted, fontsize=6, rotation=90); a.set_xlabel('FMA-UE item (standard numbering)'); a.set_ylabel('Item accuracy (%)')
a.set_ylim(0, 118); a.set_yticks([0,20,40,60,80,100]); a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True); a.legend(frameon=False, fontsize=7, loc='upper left', ncol=2)
a.set_title('A  Each item is largely predictable from the others', loc='left', fontsize=8.5, fontweight='bold')
W = np.array(H['prior2d']); g = np.linspace(-6, 6, 31)
im = b.imshow(W.T, origin='lower', extent=[-6, 6, -6, 6], cmap='Blues', aspect='auto')
b.set_xlabel(r'Proximal ability $\theta_1$ (arm items)'); b.set_ylabel(r'Distal ability $\theta_2$ (wrist/hand items)')
b.set_title('B  Learned joint ability prior', loc='left', fontsize=8.5, fontweight='bold')
for s in ('top', 'right'): b.spines[s].set_visible(True)
plt.tight_layout(); plt.savefig('fig2_hierarchy.png', dpi=300); plt.savefig('fig2_hierarchy.pdf')

# ---------------- Fig 3: controlled missingness ---------------------------------------------------------------------
S = json.load(open('res_sim.json')); ps = [0.0, 0.25, 0.5, 0.75, 1.0]
meth = [('RF-median', 'Item classifier, median imputation', '#c3c2b7', 's'), ('RF-zero', 'Item classifier, zero rule', ORANGE, '^'),
        ('Fusion-flat', 'Latent fusion, missing as uninformative', BLUE, 'o'), ('Fusion-MA', 'Missingness-aware latent fusion', AQUA, 'D')]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=False)
for ax, mech, title in ((axs[0], 'MNAR', 'A  Informative missingness (MNAR)'), (axs[1], 'MCAR', 'B  Random missingness (MCAR)')):
    for k, lab, c, mk in meth:
        y = [S[f'{mech}_{p}'][k]['mae'] for p in ps]
        ax.plot([p * 100 for p in ps], y, color=c, marker=mk, ms=5, lw=2, label=lab)
    ax.set_xlabel('Recordings removed (%)'); ax.set_ylabel('Total-score MAE (points)'); ax.set_title(title, loc='left', fontsize=8, fontweight='bold')
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xticks([0, 25, 50, 75, 100])
axs[1].set_ylim(0, 24); axs[0].set_ylim(0, 9)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=7.5)
plt.tight_layout(rect=[0, 0.12, 1, 1]); plt.savefig('fig3_missingness.png', dpi=300); plt.savefig('fig3_missingness.pdf')
print('ok')
