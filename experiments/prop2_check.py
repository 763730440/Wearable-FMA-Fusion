"""Numerical illustration of Proposition 2: evidence from the other items about item j is bounded under the latent model,
but grows with the number of items in the naive-Bayes logarithmic pool."""
import pickle, numpy as np
from scipy.special import logsumexp
from common import *
from irt import evidence
from exp_unc import tables
C = pickle.load(open('s1_rom.pkl', 'rb')); Y = YALL[RIDX]; R0 = M[RIDX].astype(bool); J = 27
res = {'latent': [], 'logpool': [], 'bound': [], 'lp_byJ': {}, 'lat_byJ': {}}
for k in range(5):
    sp = C[f'cv{k}']; tr, te, P = sp['tr'], sp['te'], sp['Pte']; g, to, tc = pickle.load(open(f'fs_model_cv{k}.pkl', 'rb'))
    lw = np.log(g.w + 1e-300); LP = g.LP                       # (J,Q,3)
    # bound B_j = max over a,b of log sup_theta p(theta|y_j=a)/p(theta|y_j=b) over the prior support
    sup = g.w > 1e-4 * g.w.max()
    for j in range(J):
        lpt = LP[j][sup]                                      # log P(y|theta) on support
        lpy = logsumexp(lw[sup][:, None] + lpt, axis=0)      # log P(y)
        lr = (lpt - lpy)                                      # log p(theta|y)/p(theta) up to w
        res['bound'].append(max((lr[:, a] - lr[:, b]).max() for a in range(3) for b in range(3) if a != b))
    T = tables(Y[tr]); cij = T[1]
    rng = np.random.default_rng(k)
    for nJ in (3, 6, 12, 26):
        lat, lp = [], []
        for n in range(len(te)):
            for j in range(J):
                others = rng.choice([i for i in range(J) if i != j], nJ, replace=False)
                lc = evidence(P[n:n + 1, others], 1.0)[0]                       # (nJ,3) untempered evidence
                A = logsumexp(LP[others] + lc[:, None, :], axis=2).sum(0)       # (Q,)
                post = lw + A; post -= logsumexp(post)
                pred = logsumexp(post[:, None] + LP[j], axis=0); prior = logsumexp(lw[:, None] + LP[j], axis=0)
                ctx = pred - prior; lat.append(ctx.max() - ctx.min())
                L = np.log(np.einsum('iab,ib->ia', cij[j, others], P[n, others] + 1e-4)).sum(0)
                lp.append(L.max() - L.min())
        res['lat_byJ'].setdefault(nJ, []).extend(lat); res['lp_byJ'].setdefault(nJ, []).extend(lp)
print('bound B_j: median %.2f, max %.2f' % (np.median(res['bound']), np.max(res['bound'])))
for nJ in (3, 6, 12, 26):
    print(nJ, 'latent context log-odds range median %.2f (95th %.2f) | log pool median %.2f (95th %.2f)' % (
        np.median(res['lat_byJ'][nJ]), np.percentile(res['lat_byJ'][nJ], 95), np.median(res['lp_byJ'][nJ]), np.percentile(res['lp_byJ'][nJ], 95)))
import json; json.dump({'bound_median': float(np.median(res['bound'])), 'bound_max': float(np.max(res['bound'])),
    'byJ': {nJ: dict(latent=float(np.median(res['lat_byJ'][nJ])), latent95=float(np.percentile(res['lat_byJ'][nJ], 95)),
                     logpool=float(np.median(res['lp_byJ'][nJ])), logpool95=float(np.percentile(res['lp_byJ'][nJ], 95))) for nJ in (3, 6, 12, 26)}},
    open('res_prop2.json', 'w'), indent=1)
