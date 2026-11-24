"""Missingness-aware latent-trait fusion.
Each item contributes two observations: whether a sensor recording exists (R_j) and, if it does, the classifier evidence.
Recording absence is modelled as depending on the item's true score (a patient who cannot perform a task produces no
usable recording):  P(R_j = 0 | y_j = k) = rho_k.  Item evidence therefore becomes
    recorded:  l_j(k) ∝ (1 - rho_k) * q_j(k)^tau        missing:  l_j(k) ∝ rho_k
In a new center, rho is re-estimated by EM from unlabeled target assessments (posterior expected item scores of missing
vs recorded items under the fixed measurement model); clinician scores of the target center are never used."""
import numpy as np
from scipy.special import logsumexp

def evidence_m(Pq, tau, R, rho):
    le = tau * np.log(Pq + 1e-4) + np.log(1 - rho + 1e-9)[None, None]
    lm = np.broadcast_to(np.log(rho + 1e-9)[None, None], le.shape)
    le = np.where(R[..., None], le, lm)
    return le - logsumexp(le, axis=2, keepdims=True)

def fuse_m(model, Pq, R, rho, tau_o=1.0, tau_c=1.0, w=None):
    w = model.w if w is None else w
    lo = evidence_m(Pq, tau_o, R, rho); lc = evidence_m(Pq, tau_c, R, rho)
    A = logsumexp(model.LP[None] + lc[:, :, None, :], axis=3)
    lpost = np.log(w + 1e-300)[None] + A.sum(1)
    loo = lpost[:, None, :] - A; loo = loo - logsumexp(loo, axis=2, keepdims=True)
    pred = logsumexp(loo[..., None] + model.LP[None], axis=2)
    item = lo + pred; item = np.exp(item - logsumexp(item, axis=2, keepdims=True))
    return item

def rho_from_labels(Y, R, a=0.5):
    """Source-domain estimate with a Beta(a, a) smoothing."""
    return np.array([((~R) & (Y == k)).sum() + a for k in range(3)]) / np.array([(Y == k).sum() + 2 * a for k in range(3)])

def estimate_rho(model, Pq, R, rho0, tau_o, tau_c, iters=30, a=0.5):
    """Unsupervised EM on target assessments."""
    rho = rho0.copy()
    for _ in range(iters):
        item = fuse_m(model, Pq, R, rho, tau_o, tau_c)
        num = np.array([(item[..., k] * (~R)).sum() + a for k in range(3)]); den = np.array([item[..., k].sum() + 2 * a for k in range(3)])
        rho = num / den
    return rho
