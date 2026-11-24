"""Learned cross-item fusion baselines operating on the same item evidence as the latent-trait fusion.
Each assessment is a set of 27 item tokens: [log class probabilities (3), recording flag (1)] + a learned item embedding.
  ItemTransformer : 2 self-attention blocks (pre-norm, 4 heads, d = 32) over the item tokens
  ItemGCN         : 2 graph-convolution blocks over an item graph built from training clinician scores
                    (k = 6 nearest items by absolute Spearman correlation, symmetric, self-loops, degree-normalized)
Both output per-item 3-class logits, are trained on inner out-of-fold classifier outputs of the training patients with
cross-entropy and AdamW, early-stopped on a 20% patient-level validation split, and averaged over 3 seeds."""
import sys; sys.path.insert(0, "work")
import numpy as np
from scipy.stats import spearmanr
from nn import T, Adam, concat
from nnx import transpose, softmax, layernorm, cross_entropy, param

J, K = 27, 3

def tokens(P, R):
    return np.concatenate([np.log(np.where(R[..., None], P, 1 / 3) + 1e-4), R[..., None].astype(np.float32)], -1).astype(np.float32)

class ItemTransformer:
    def __init__(self, d=32, heads=4, layers=2, seed=0):
        rng = np.random.default_rng(seed); self.d, self.h, self.L = d, heads, layers
        self.Win = param(4, d, rng=rng); self.bin = param(1, d, scale=0.0, rng=rng); self.E = param(J, d, scale=0.1, rng=rng)
        self.blocks = []
        for _ in range(layers):
            b = dict(g1=T(np.ones(d), req=True), b1=T(np.zeros(d), req=True), Wq=param(d, d, rng=rng), Wk=param(d, d, rng=rng), Wv=param(d, d, rng=rng),
                     Wo=param(d, d, scale=0.5 / np.sqrt(d), rng=rng), g2=T(np.ones(d), req=True), b2=T(np.zeros(d), req=True),
                     W1=param(d, 2 * d, rng=rng), c1=T(np.zeros(2 * d), req=True), W2=param(2 * d, d, scale=0.5 / np.sqrt(2 * d), rng=rng), c2=T(np.zeros(d), req=True))
            self.blocks.append(b)
        self.gf = T(np.ones(d), req=True); self.bf = T(np.zeros(d), req=True); self.Wout = param(d, K, rng=rng); self.bout = T(np.zeros(K), req=True)
    def params(self):
        p = [self.Win, self.bin, self.E, self.gf, self.bf, self.Wout, self.bout]
        for b in self.blocks: p += list(b.values())
        return p
    def forward(self, X):
        N = X.shape[0]; h = T(X) @ self.Win + self.bin + self.E; dh = self.d // self.h
        for b in self.blocks:
            z = layernorm(h, b['g1'], b['b1']); q, k, v = z @ b['Wq'], z @ b['Wk'], z @ b['Wv']; heads = []
            for i in range(self.h):
                sl = (slice(None), slice(None), slice(i * dh, (i + 1) * dh))
                a = softmax((q[sl] @ transpose(k[sl])) * (1 / np.sqrt(dh))); heads.append(a @ v[sl])
            h = h + concat(heads, -1) @ b['Wo']
            z = layernorm(h, b['g2'], b['b2']); h = h + ((z @ b['W1'] + b['c1']).relu() @ b['W2'] + b['c2'])
        return layernorm(h, self.gf, self.bf) @ self.Wout + self.bout

class ItemGCN:
    def __init__(self, A, d=32, layers=2, seed=0):
        rng = np.random.default_rng(seed); self.A = A.astype(np.float32)
        self.Win = param(4, d, rng=rng); self.bin = param(1, d, scale=0.0, rng=rng); self.E = param(J, d, scale=0.1, rng=rng)
        self.W = [param(d, d, scale=0.5 / np.sqrt(d), rng=rng) for _ in range(layers)]; self.b = [T(np.zeros(d), req=True) for _ in range(layers)]
        self.g = [T(np.ones(d), req=True) for _ in range(layers)]; self.be = [T(np.zeros(d), req=True) for _ in range(layers)]
        self.Wout = param(d, K, rng=rng); self.bout = T(np.zeros(K), req=True)
    def params(self): return [self.Win, self.bin, self.E, self.Wout, self.bout] + self.W + self.b + self.g + self.be
    def forward(self, X):
        h = (T(X) @ self.Win + self.bin + self.E).relu(); At = T(self.A)
        for W, b, g, be in zip(self.W, self.b, self.g, self.be):
            h = h + (At @ layernorm(h, g, be) @ W + b).relu()
        return h @ self.Wout + self.bout

def item_graph(Y, k=6):
    C = np.abs(spearmanr(Y).correlation); np.fill_diagonal(C, 0); C = np.nan_to_num(C)
    A = np.zeros_like(C)
    for j in range(J): A[j, np.argsort(C[j])[-k:]] = 1
    A = np.maximum(A, A.T) + np.eye(J); d = A.sum(1); return A / np.sqrt(np.outer(d, d))

def _train_one(make, X, Y, grp, seed, epochs=400, lr=3e-3, wd=1e-3, patience=40):
    rng = np.random.default_rng(seed); ug = np.unique(grp); val_g = rng.choice(ug, max(1, len(ug) // 5), replace=False)
    va = np.isin(grp, val_g); tr = ~va
    m = make(seed); opt = Adam(m.params(), lr=lr, wd=wd); best = (np.inf, None, 0)
    for ep in range(epochs):
        loss = cross_entropy(m.forward(X[tr]), Y[tr]); loss.backward(); opt.step()
        vl = cross_entropy(m.forward(X[va]), Y[va]).v.item()
        if vl < best[0] - 1e-4: best = (vl, [p.v.copy() for p in m.params()], ep)
        elif ep - best[2] > patience: break
    for p, v in zip(m.params(), best[1]): p.v = v
    return m

def fit_predict(kind, P_oof, R_tr, Y_tr, grp_tr, P_te, R_te, seeds=(0, 1, 2)):
    Xtr = tokens(P_oof, R_tr); Xte = tokens(P_te, R_te); out = 0
    if kind == 'transformer': make = lambda s: ItemTransformer(seed=s)
    else:
        A = item_graph(Y_tr); make = lambda s: ItemGCN(A, seed=s)
    for s in seeds:
        m = _train_one(make, Xtr, Y_tr, grp_tr, s); lg = m.forward(Xte).v
        e = np.exp(lg - lg.max(-1, keepdims=True)); out = out + e / e.sum(-1, keepdims=True)
    return out / len(seeds)
