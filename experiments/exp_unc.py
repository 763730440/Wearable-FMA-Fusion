"""Uncertainty, classical fusion baselines, sensor-block dropout and extra ablations (Xi'an CV + center transfer).
Outputs unc_preds_<split>.pkl (item probabilities and total-score distributions per method) and is summarised by
sum_unc.py. Classical fusion baselines use the same item evidence and pairwise score dependencies estimated from
training labels: linear opinion pool, logarithmic pool (naive-Bayes product of experts), Dempster-Shafer with discounting."""
import pickle, sys, os, numpy as np
from scipy.special import logsumexp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import *
from exp_a import ctx_feats, rf
from irt import fuse, fit_tau, evidence
from mirt import MGPCM, TH, DIM
from missirt import rho_from_labels, estimate_rho, evidence_m

C = pickle.load(open('s1_rom.pkl', 'rb')); r = RIDX; Y = YALL[r]; R0 = M[r].astype(bool); J = len(ITEMS)
EPS = 1e-4

def stage2_proba(oof, ytr, Pte):
    Ftr = ctx_feats(oof); Fte = ctx_feats(Pte); out = np.zeros(Pte.shape)
    for j in range(J):
        y = ytr[:, j]
        if len(np.unique(y)) == 1: out[:, j, y[0]] = 1; continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Ftr[j], y)
        out[:, j][:, m.classes_] = m.predict_proba(Fte[j])
    return np.clip(out, 1e-6, 1)

# ---------- classical pools ----------------------------------------------------------------------------------------
def tables(Ytr):
    oh = np.eye(3)[Ytr]; N = np.einsum('nja,nib->jiab', oh, oh) + 1.0         # N[j,i,a,b] = #(y_j=a, y_i=b)
    pi = oh.mean(0) + 1e-3; pi /= pi.sum(1, keepdims=True)
    cond_i_given_j = N / N.sum(3, keepdims=True)                                 # P(y_i=b | y_j=a)
    cond_j_given_i = N / N.sum(2, keepdims=True)                                 # P(y_j=a | y_i=b)
    return pi, cond_i_given_j, cond_j_given_i

def item_post(P, pi, mask):        # P(y_i | x_i) from balanced-class evidence and the training prior; missing -> prior
    q = pi[None] * (P + EPS); q /= q.sum(2, keepdims=True)
    return np.where(mask[..., None], q, pi[None])

def lin_pool(P, mask, T, w):
    pi, cij, cji = T; q = item_post(P, pi, mask)
    msg = np.einsum('jiab,nib->njia', cji, q)                                    # P(y_j=a | x_i)
    off = ~np.eye(J, dtype=bool); m = mask[:, None, :] & off[None]               # use recorded other items only
    other = (msg * m[..., None]).sum(2) / np.maximum(m.sum(2), 1)[..., None]
    own = np.where(mask[..., None], q, other)
    return w * own + (1 - w) * other

def log_pool(P, mask, T, to, tc):
    pi, cij, cji = T
    L = np.einsum('jiab,nib->njia', cij, P + EPS)                                # p(x_i | y_j=a) up to const
    off = ~np.eye(J, dtype=bool); m = mask[:, None, :] & off[None]
    lg = np.log(pi)[None] + to * np.where(mask[..., None], np.log(P + EPS), 0) + tc * (np.log(L) * m[..., None]).sum(2)
    return np.exp(lg - logsumexp(lg, axis=2, keepdims=True))

def ds_pool(P, mask, T, a0, a):
    pi, cij, cji = T; q = item_post(P, pi, mask)
    msg = np.einsum('jiab,nib->njia', cji, q)                                    # (n, j, i, a)
    off = ~np.eye(J, dtype=bool); m = (mask[:, None, :] & off[None])[..., None]
    lA = (np.log(a * msg + (1 - a)) * m).sum(2); lB = (np.log(1 - a) * m[..., 0]).sum(2)      # others
    own_k = np.where(mask[..., None], a0 * q, 0.0); own_t = np.where(mask, 1 - a0, 1.0)
    lA = lA + np.log(own_k + own_t[..., None]); lB = lB + np.log(own_t)
    s = np.exp(lA - lA.max(2, keepdims=True)); t = np.exp(lB - lA.max(2))
    mk = np.maximum(s - t[..., None], 0); bet = mk + t[..., None] / 3                     # pignistic transform
    return bet / bet.sum(2, keepdims=True)

def nll(p, y): return -np.log(np.take_along_axis(p, y[..., None], 2) + 1e-9).mean()

def tune_pools(oof, ytr, mtr):
    T = tables(ytr); G = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    w = min((0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0), key=lambda w: nll(lin_pool(oof, mtr, T, w), ytr))
    tt = min(((a, b) for a in G for b in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)), key=lambda z: nll(log_pool(oof, mtr, T, *z), ytr))
    ds = min(((a, b) for a in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99) for b in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)), key=lambda z: nll(ds_pool(oof, mtr, T, *z), ytr))
    return T, w, tt, ds

# ---------- total-score distributions -----------------------------------------------------------------------------
def conv_indep(p):           # p (n, J, 3) -> (n, 2J+1)
    d = np.zeros((p.shape[0], 2 * J + 1)); d[:, 0] = 1
    for j in range(J):
        nd = d * p[:, j, :1]; nd[:, 1:] += d[:, :-1] * p[:, j, 1:2]; nd[:, 2:] += d[:, :-2] * p[:, j, 2:3]; d = nd
    return d

def latent_total(LP, lw, lo, lpost_c):
    """Joint latent-trait posterior of the total: sum_theta p(theta|x) * conv_j p(y_j | theta, x_j)."""
    out = []
    for n in range(lo.shape[0]):
        c = LP + lo[n][:, None, :]; c = np.exp(c - logsumexp(c, axis=2, keepdims=True))     # (J, Q, 3)
        d = np.zeros((LP.shape[1], 2 * J + 1)); d[:, 0] = 1
        for j in range(J):
            nd = d * c[j, :, :1]; nd[:, 1:] += d[:, :-1] * c[j, :, 1:2]; nd[:, 2:] += d[:, :-2] * c[j, :, 2:3]; d = nd
        w = np.exp(lpost_c[n] - logsumexp(lpost_c[n])); out.append(w @ d)
    return np.array(out)

def latent_parts(g, P, to, tc, mask, w=None):
    w = g.w if w is None else w; lo = evidence(P, to, mask); lc = evidence(P, tc, mask)
    A = logsumexp(g.LP[None] + lc[:, :, None, :], axis=3); lpost = np.log(w + 1e-300)[None] + A.sum(1)
    return lo, lpost

def gauss_prior(g):
    mu = g.w @ TH; X = TH - mu; S = (g.w[:, None] * X).T @ X; Si = np.linalg.inv(S)
    w = np.exp(-0.5 * np.einsum('qa,ab,qb->q', X, Si, X)); return w / w.sum()

# ---------- per split --------------------------------------------------------------------------------------------
def run(split):
    sp = C[split]; tr, te, P, oof = sp['tr'], sp['te'], sp['Pte'], sp['oof']; Rt = R0[te]
    out = dict(te=te, prob={}, tot={}, pred={})
    g = MGPCM(smooth=1.0).fit(Y[tr]); to, tc = fit_tau(g, oof, Y[tr], mask=R0[tr])
    T, w, tt, ds = tune_pools(oof, Y[tr], R0[tr]); out['params'] = dict(to=to, tc=tc, w=w, logpool=tt, ds=ds)
    def add(name, prob, tot=None):
        out['prob'][name] = prob; out['pred'][name] = prob.argmax(2); out['tot'][name] = conv_indep(prob) if tot is None else tot
    pi = T[0]
    add('RF', item_post(P, pi, Rt))
    add('Stacking', stage2_proba(oof, Y[tr], P))
    add('Linear opinion pool', lin_pool(P, Rt, T, w))
    add('Log pool (product of experts)', log_pool(P, Rt, T, *tt))
    add('Dempster-Shafer', ds_pool(P, Rt, T, *ds))
    item, _ = fuse(g, P, to, tc, mask=Rt); lo, lpost = latent_parts(g, P, to, tc, Rt)
    add('Latent fusion', item, latent_total(g.LP, g.w, lo, lpost))
    add('Latent fusion + Stacking (average)', 0.5 * (item + out['prob']['Stacking']))
    # ablations
    g0 = MGPCM(smooth=0.0).fit(Y[tr]); a, b = fit_tau(g0, oof, Y[tr], mask=R0[tr]); out['pred']['Abl: unsmoothed prior'] = fuse(g0, P, a, b, mask=Rt)[0].argmax(2)
    out['pred']['Abl: Gaussian prior'] = fuse(g, P, to, tc, w=gauss_prior(g), mask=Rt)[0].argmax(2)
    # sensor-block dropout (device failure: not score-related) on test assessments
    med = np.zeros(J, int)
    for j, it in enumerate(ITEMS):
        X = D.iloc[r[tr]][FC[it]].to_numpy(float); y = Y[tr, j]
        med[j] = y[0] if len(np.unique(y)) == 1 else rf().fit(X, y).predict(np.full((1, X.shape[1]), np.nan))[0]
    rho0 = rho_from_labels(Y[tr], R0[tr]); out['block'] = {}
    for blk in (0, 1):
        Rb = Rt & (DIM != blk)[None]; Pu = np.where(Rb[..., None], P, 1 / 3)
        b = {'RF (median imputation)': np.where(Rb, P.argmax(2), med[None]),
             'Stacking': stage2_proba(oof, Y[tr], Pu).argmax(2),
             'Linear opinion pool': lin_pool(P, Rb, T, w).argmax(2),
             'Log pool (product of experts)': log_pool(P, Rb, T, *tt).argmax(2),
             'Dempster-Shafer': ds_pool(P, Rb, T, *ds).argmax(2),
             'Latent fusion (block flagged as device failure)': fuse(g, P, to, tc, mask=Rb)[0].argmax(2)}
        from missirt import fuse_m
        b['Missingness-aware fusion (block not flagged)'] = fuse_m(g, P, Rb, rho0, to, tc).argmax(2)
        out['block']['glove' if blk == 1 else 'arm'] = b
    pickle.dump(out, open(f'unc_{split}.pkl', 'wb')); print(split, 'done', out['params'], flush=True)

if __name__ == '__main__':
    for s in sys.argv[1:]: run(s)
