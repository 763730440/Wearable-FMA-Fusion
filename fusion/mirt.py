"""Two-dimensional (proximal arm / distal wrist-hand) between-item GPCM for latent-trait fusion.
Arm-sensor items (shoulder, elbow, forearm, coordination) load on theta_1; glove items (wrist, hand) on theta_2.
The joint population distribution w(theta_1, theta_2) is nonparametric on a grid, so the dependence between proximal
and distal recovery is learned from data rather than assumed."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, log_softmax
from irt import item_logp
from common import ITEMS
g1 = np.linspace(-6, 6, 31)
T1, T2 = np.meshgrid(g1, g1, indexing='ij'); TH = np.c_[T1.ravel(), T2.ravel()]       # (Q, 2)
DIM = np.array([0 if (it <= 17 or it == 31) else 1 for it in ITEMS])

class MGPCM:
    def __init__(self, lam=0.05, iters=40, smooth=0.0):
        self.lam, self.iters, self.smooth = lam, iters, smooth
    def _LP(self):
        return np.stack([item_logp(self.a[j], self.d[j], grid=TH[:, DIM[j]]) for j in range(len(DIM))])
    def fit(self, Y):
        n, J = Y.shape
        self.a = np.zeros(J); self.d = np.zeros((J, 2))
        for j in range(J):
            f = np.bincount(Y[:, j], minlength=3) + 0.5; self.d[j] = [np.log(f[0] / f[1]), np.log(f[1] / f[2])]
        w = np.exp(-(TH ** 2).sum(1) / 2 + 0.6 * TH[:, 0] * TH[:, 1] / 2); self.w = w / w.sum()
        oh = np.eye(3)[Y]
        for _ in range(self.iters):
            LP = self._LP(); ll = np.einsum('njk,jqk->nq', oh, LP)
            post = np.exp(log_softmax(np.log(self.w + 1e-300)[None] + ll, axis=1))
            N = np.einsum('nq,njk->jqk', post, oh)
            for j in range(J):
                gj = TH[:, DIM[j]]
                f = lambda p: -(N[j] * item_logp(p[0], p[1:], grid=gj)).sum() + self.lam * (p[0] ** 2 + (p[1:] ** 2).sum())
                rr = minimize(f, np.r_[self.a[j], self.d[j]], method='L-BFGS-B'); self.a[j], self.d[j] = rr.x[0], rr.x[1:]
            w = post.mean(0) + 1e-7
            if self.smooth:
                W = w.reshape(31, 31); from scipy.ndimage import gaussian_filter; w = gaussian_filter(W, self.smooth).ravel()
            self.w = w / w.sum()
        self.LP = self._LP(); return self
