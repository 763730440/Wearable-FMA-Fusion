"""Latent-trait (IRT) evidence fusion for item-level FMA scoring.

Measurement model: generalized partial credit model (GPCM) for each item j with categories k = 0,1,2
    P(y_j = k | theta) = softmax_k( a_j * (k*theta - sum_{h<=k} d_jh) )
fitted by marginal maximum likelihood (EM) on clinician item scores of the training patients, with a
nonparametric (grid) population distribution w(theta).

Fusion: each item classifier provides evidence l_j(k) (tempered class probabilities; classifiers are trained with
balanced class weights, so their outputs act as likelihoods under a flat prior). The latent ability posterior combines
the evidence of all items through the measurement model; each item's final posterior combines its own evidence with
the leave-one-out ability posterior of the other items (no double counting):
    p(theta | x_-j)  ∝ w(theta) * prod_{i != j} sum_k P(y_i = k | theta) l_i(k)^tau_c
    p(y_j = k | x)   ∝ l_j(k)^tau_o * sum_theta P(y_j = k | theta) p(theta | x_-j)
Domain adaptation: with item parameters fixed (measurement invariance), the target population distribution w(theta)
is re-estimated by EM from unlabeled target evidence."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, log_softmax

GRID = np.linspace(-6, 6, 61)

def item_logp(a, d, grid=GRID):
    """a: scalar log-discrimination, d: (2,) step parameters -> (Q, 3) log-probabilities."""
    c = np.r_[0.0, np.cumsum(d)]
    s = np.exp(a) * (np.arange(3)[None, :] * grid[:, None] - c[None, :])
    return log_softmax(s, axis=1)

class GPCM:
    def __init__(self, lam=0.05, iters=40):
        self.lam, self.iters = lam, iters

    def fit(self, Y):
        n, J = Y.shape; Q = len(GRID)
        self.a = np.zeros(J); self.d = np.zeros((J, 2))
        # init steps from category frequencies
        for j in range(J):
            f = np.bincount(Y[:, j], minlength=3) + 0.5
            self.d[j] = [np.log(f[0] / f[1]), np.log(f[1] / f[2])]
        self.w = np.exp(-GRID ** 2 / 2); self.w /= self.w.sum()
        onehot = np.eye(3)[Y]                                         # (n, J, 3)
        for it in range(self.iters):
            LP = np.stack([item_logp(self.a[j], self.d[j]) for j in range(J)])     # (J, Q, 3)
            ll = np.einsum('njk,jqk->nq', onehot, LP)                               # (n, Q)
            post = np.exp(log_softmax(np.log(self.w + 1e-300)[None] + ll, axis=1))  # (n, Q)
            N = np.einsum('nq,njk->jqk', post, onehot)                              # expected counts
            for j in range(J):
                def f(p):
                    return -(N[j] * item_logp(p[0], p[1:])).sum() + self.lam * (p[0] ** 2 + (p[1:] ** 2).sum())
                r = minimize(f, np.r_[self.a[j], self.d[j]], method='L-BFGS-B')
                self.a[j], self.d[j] = r.x[0], r.x[1:]
            self.w = post.mean(0) + 1e-6; self.w /= self.w.sum()
        self.LP = np.stack([item_logp(self.a[j], self.d[j]) for j in range(J)])
        return self

def evidence(Pq, tau, mask=None):
    """Pq: (n, J, 3) classifier probabilities -> tempered log-evidence; missing items -> flat."""
    le = tau * np.log(Pq + 1e-4)
    if mask is not None: le = np.where(mask[..., None], le, 0.0)
    return le - logsumexp(le, axis=2, keepdims=True)

def fuse(model, Pq, tau_o=1.0, tau_c=1.0, w=None, mask=None):
    """Returns item posteriors (n, J, 3) and ability posterior (n, Q)."""
    w = model.w if w is None else w
    lo = evidence(Pq, tau_o, mask); lc = evidence(Pq, tau_c, mask)
    A = logsumexp(model.LP[None] + lc[:, :, None, :], axis=3)               # (n, J, Q): log sum_k P(k|theta) l(k)
    lpost = np.log(w + 1e-300)[None] + A.sum(1)                              # (n, Q)
    loo = lpost[:, None, :] - A                                              # (n, J, Q)
    loo = loo - logsumexp(loo, axis=2, keepdims=True)
    pred = logsumexp(loo[..., None] + model.LP[None], axis=2)                # (n, J, 3): log sum_theta P(k|theta) p(theta|x_-j)
    item = lo + pred
    item = np.exp(item - logsumexp(item, axis=2, keepdims=True))
    return item, np.exp(lpost - logsumexp(lpost, axis=1, keepdims=True))

def adapt_prior(model, Pq, tau_c=1.0, mask=None, iters=50, smooth=0.0):
    """Unsupervised EM re-estimation of the population ability distribution on target evidence."""
    w = model.w.copy()
    for _ in range(iters):
        _, post = fuse(model, Pq, 1.0, tau_c, w, mask)
        w_new = post.mean(0)
        if smooth: w_new = (1 - smooth) * w_new + smooth * model.w
        w = w_new / w_new.sum()
    return w

def fit_tau(model, Poof, Y, grid=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0), mask=None):
    """Choose (tau_o, tau_c) maximizing the log-likelihood of true item scores on inner out-of-fold evidence."""
    best = None
    for to in grid:
        for tc in grid:
            item, _ = fuse(model, Poof, to, tc, mask=mask)
            ll = np.log(np.take_along_axis(item, Y[..., None], 2) + 1e-9).sum()
            if best is None or ll > best[0]: best = (ll, to, tc)
    return best[1], best[2]

def fuse_w(model, Pq, tau_o, tau_c_vec, w=None, mask=None):
    """Like fuse(), but with an item-specific reliability exponent tau_c_j for the evidence entering the ability posterior.
    Also returns the leave-one-out predictive distribution p(y_j | x_-j)."""
    w = model.w if w is None else w
    lo = evidence(Pq, tau_o, mask)
    lc = np.stack([evidence(Pq[:, j:j + 1], tau_c_vec[j], None if mask is None else mask[:, j:j + 1])[:, 0] for j in range(Pq.shape[1])], 1)
    A = logsumexp(model.LP[None] + lc[:, :, None, :], axis=3)
    lpost = np.log(w + 1e-300)[None] + A.sum(1)
    loo = lpost[:, None, :] - A; loo = loo - logsumexp(loo, axis=2, keepdims=True)
    pred = logsumexp(loo[..., None] + model.LP[None], axis=2)
    pred = pred - logsumexp(pred, axis=2, keepdims=True)
    item = lo + pred; item = np.exp(item - logsumexp(item, axis=2, keepdims=True))
    return item, np.exp(pred)

def fit_tau_items(model, Poof, Y, mask=None, grid=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0), passes=2):
    """Reliability-aware fusion: global own-evidence exponent and item-specific context exponents, by coordinate ascent
    on the inner out-of-fold log-likelihood of true item scores."""
    J = Poof.shape[1]; to = 1.5; tc = np.full(J, 2.0)
    def ll(to, tc):
        item, _ = fuse_w(model, Poof, to, tc, mask=mask)
        return np.log(np.take_along_axis(item, Y[..., None], 2) + 1e-9).sum()
    for _ in range(passes):
        to = max(grid[1:], key=lambda t: ll(t, tc))
        for j in range(J):
            best = max(grid, key=lambda t: ll(to, np.r_[tc[:j], t, tc[j + 1:]])); tc[j] = best
    return to, tc
