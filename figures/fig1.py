"""Fig 1: framework schematic of missingness-aware latent-trait fusion."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})
INK = '#0b0b0b'; SUB = '#52514e'; BLUE = '#2a78d6'; ORANGE = '#eb6834'; AQUA = '#1baf7a'; GRAY = '#e9e8e4'
fig, ax = plt.subplots(figsize=(9.0, 4.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis('off')

def box(x, y, w, h, text, fc='white', ec=SUB, lw=0.8, fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK, fontweight='bold' if bold else 'normal', linespacing=1.25)

def arrow(x1, y1, x2, y2, c=SUB):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=8, lw=0.8, color=c))

# column titles
for x, t in ((9, '1  Task-level sensing'), (33, '2  Item evidence'), (60, '3  Latent-trait fusion'), (88, '4  Output')):
    ax.text(x, 48, t, ha='center', va='center', fontsize=8.5, fontweight='bold', color=INK)
# sensing
box(1, 32, 16, 10, 'IMU (upper arm,\nforearm)\nitems 3-17, 31', fc='#eaf2fc', ec=BLUE)
box(1, 18, 16, 10, 'Sensor glove\nwrist & hand\nitems 19-30', fc='#fdeee7', ec=ORANGE)
box(1, 5, 16, 9, 'No recording\n(task not\nperformed)', fc=GRAY)
# evidence
box(24, 30, 18, 13, 'Item classifier\n(kinematic features)\n$q_j(k)$, k = 0, 1, 2')
box(24, 14, 18, 11, 'Recording\nindicator $R_j$\n$P(R_j{=}0\\mid y_j{=}k)=\\rho_k$')
box(24, 3, 18, 7.5, '$\\rho$ re-estimated by EM\non unlabeled target data', fc=GRAY, fs=7.5)
for y0 in (37, 23): arrow(17.6, y0, 23.4, 36.5 if y0 == 37 else 34)
arrow(17.6, 9.5, 23.4, 18)
# fusion
box(49, 27, 22, 16, 'Two-dimensional GPCM\n$P(y_j{=}k\\mid\\theta_{d(j)})$\nproximal $\\theta_1$ | distal $\\theta_2$\njoint prior $w(\\theta_1,\\theta_2)$', fc='#e8f7f1', ec=AQUA, lw=1.1)
box(49, 8, 22, 14, 'Item posterior\n$p(y_j\\mid x)\\propto \\ell_j(k)^{\\tau_o}$\n$\\times\\sum_\\theta P(y_j{=}k\\mid\\theta)\\,p(\\theta\\mid x_{-j})$', fc='#e8f7f1', ec=AQUA, lw=1.1)
arrow(42.6, 36.5, 48.4, 36.5); arrow(42.6, 19.5, 48.4, 31); arrow(60, 26.4, 60, 22.6)
ax.text(60, 4.0, 'leave-one-out ability posterior: no double counting', fontsize=6.5, color=SUB, ha='center', va='center')
# output
box(80, 27, 17, 14, 'Item scores\n(27 items, 0-2)', bold=False)
box(80, 8, 17, 14, 'Instrumented total\n+ therapist rating\nof 6 items', bold=False)
arrow(71.6, 14.5, 79.4, 33); arrow(88.5, 26.4, 88.5, 22.6)
plt.savefig('fig1_framework.png', dpi=300, bbox_inches='tight'); plt.savefig('fig1_framework.pdf', bbox_inches='tight'); print('ok')
