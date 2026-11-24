import json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
R = json.load(open('results_paper.json'))['beijing']
rows = [('RF', 'RF (median imputation)'), ('RF + zero rule', 'RF + zero rule'), ('HT-Net', 'HT-Net (masking)'), ('Stacking', 'Stacking'),
        ('Stacking + zero rule', 'Stacking + zero rule'), ('IRT-2D', 'Latent fusion (missing uninformative)'), ('MA-IRT', 'Missingness-aware latent fusion')]
BLUE, ORANGE = '#2a78d6', '#eb6834'
fig, ax = plt.subplots(figsize=(6.4, 3.0)); y = np.arange(len(rows))[::-1]
for yi, (k, lab) in zip(y, rows):
    s, m = R[k]['sev_bias'], R[k]['mild_bias']
    ax.plot([s, m], [yi, yi], color='#c3c2b7', lw=1.2, zorder=1)
    ax.scatter(s, yi, color=ORANGE, marker='^', s=36, zorder=2); ax.scatter(m, yi, color=BLUE, marker='o', s=30, zorder=2)
ax.axvline(0, color='#52514e', lw=0.8); ax.set_yticks(y); ax.set_yticklabels([l for _, l in rows])
ax.set_xlabel('Mean bias of the instrumented total, points (prediction − clinician)'); ax.xaxis.grid(True, color='#e9e8e4', lw=0.6); ax.set_axisbelow(True)
n = json.load(open('results_paper.json'))['beijing_missing']
ax.scatter([], [], color=ORANGE, marker='^', label=f"Moderate to severe (FMA27 < 40, n = {n['n_sev']})"); ax.scatter([], [], color=BLUE, marker='o', label=f"Mild (n = {n['n'] - n['n_sev']})")
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
plt.tight_layout(); plt.savefig('fig4_beijing_bias.png', dpi=300); plt.savefig('fig4_beijing_bias.pdf'); print('ok')
