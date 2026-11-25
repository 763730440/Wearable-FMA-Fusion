"""Learned cross-item fusion baselines operating on the same item evidence as the latent-trait fusion (PyTorch).
Each assessment is a set of 27 item tokens: [log class probabilities (3), recording flag (1)] + a learned item embedding.
  ItemTransformer : 2 self-attention blocks (pre-norm, 4 heads, d = 32) over the item tokens
  ItemGCN         : 2 graph-convolution blocks over an item graph built from training clinician scores
                    (k = 6 nearest items by absolute Spearman correlation, symmetric, self-loops, degree-normalized)
Both output per-item 3-class logits, are trained full-batch on inner out-of-fold classifier outputs of the training
patients with cross-entropy and Adam (L2 weight decay), early-stopped on a 20% patient-level validation split,
and averaged over 3 seeds."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

J, K = 27, 3


def tokens(P, R):
    return np.concatenate([np.log(np.where(R[..., None], P, 1 / 3) + 1e-4), R[..., None].astype(np.float32)], -1).astype(np.float32)


def _param(rng, *shape, scale=None):
    """Normal(0, scale) weights from a NumPy generator (default scale 1/sqrt(fan_in)), reproducible across platforms."""
    scale = scale if scale is not None else 1 / np.sqrt(shape[0])
    return nn.Parameter(torch.tensor(rng.normal(0, scale, shape), dtype=torch.float32))


def _ones(d): return nn.Parameter(torch.ones(d))
def _zeros(d): return nn.Parameter(torch.zeros(d))


class ItemTransformer(nn.Module):
    def __init__(self, d=32, heads=4, layers=2, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed); self.d, self.h = d, heads
        self.Win = _param(rng, 4, d); self.bin = _param(rng, 1, d, scale=0.0); self.E = _param(rng, J, d, scale=0.1)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            b = nn.ParameterDict(dict(
                g1=_ones(d), b1=_zeros(d), Wq=_param(rng, d, d), Wk=_param(rng, d, d), Wv=_param(rng, d, d),
                Wo=_param(rng, d, d, scale=0.5 / np.sqrt(d)), g2=_ones(d), b2=_zeros(d),
                W1=_param(rng, d, 2 * d), c1=_zeros(2 * d), W2=_param(rng, 2 * d, d, scale=0.5 / np.sqrt(2 * d)), c2=_zeros(d)))
            self.blocks.append(b)
        self.gf = _ones(d); self.bf = _zeros(d); self.Wout = _param(rng, d, K); self.bout = _zeros(K)

    def forward(self, X):
        X = torch.as_tensor(X); N = X.shape[0]; dh = self.d // self.h
        h = X @ self.Win + self.bin + self.E
        for b in self.blocks:
            z = F.layer_norm(h, (self.d,), b['g1'], b['b1'], eps=1e-5)
            q, k, v = (z @ b['Wq']).view(N, J, self.h, dh).transpose(1, 2), (z @ b['Wk']).view(N, J, self.h, dh).transpose(1, 2), \
                      (z @ b['Wv']).view(N, J, self.h, dh).transpose(1, 2)
            a = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(dh), -1)          # (N, heads, J, J)
            h = h + (a @ v).transpose(1, 2).reshape(N, J, self.d) @ b['Wo']
            z = F.layer_norm(h, (self.d,), b['g2'], b['b2'], eps=1e-5)
            h = h + (torch.relu(z @ b['W1'] + b['c1']) @ b['W2'] + b['c2'])
        return F.layer_norm(h, (self.d,), self.gf, self.bf, eps=1e-5) @ self.Wout + self.bout


class ItemGCN(nn.Module):
    def __init__(self, A, d=32, layers=2, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed); self.d = d
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.Win = _param(rng, 4, d); self.bin = _param(rng, 1, d, scale=0.0); self.E = _param(rng, J, d, scale=0.1)
        self.W = nn.ParameterList([_param(rng, d, d, scale=0.5 / np.sqrt(d)) for _ in range(layers)])
        self.b = nn.ParameterList([_zeros(d) for _ in range(layers)])
        self.g = nn.ParameterList([_ones(d) for _ in range(layers)]); self.be = nn.ParameterList([_zeros(d) for _ in range(layers)])
        self.Wout = _param(rng, d, K); self.bout = _zeros(K)

    def forward(self, X):
        h = torch.relu(torch.as_tensor(X) @ self.Win + self.bin + self.E)
        for W, b, g, be in zip(self.W, self.b, self.g, self.be):
            h = h + torch.relu(self.A @ F.layer_norm(h, (self.d,), g, be, eps=1e-5) @ W + b)
        return h @ self.Wout + self.bout


def item_graph(Y, k=6):
    C = np.abs(spearmanr(Y).correlation); np.fill_diagonal(C, 0); C = np.nan_to_num(C)
    A = np.zeros_like(C)
    for j in range(J): A[j, np.argsort(C[j])[-k:]] = 1
    A = np.maximum(A, A.T) + np.eye(J); d = A.sum(1); return A / np.sqrt(np.outer(d, d))


def _ce(logits, y):
    return F.cross_entropy(logits.reshape(-1, K), torch.as_tensor(y, dtype=torch.long).reshape(-1))


def _train_one(make, X, Y, grp, seed, epochs=400, lr=3e-3, wd=1e-3, patience=40):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed); ug = np.unique(grp); val_g = rng.choice(ug, max(1, len(ug) // 5), replace=False)
    va = np.isin(grp, val_g); tr = ~va
    m = make(seed); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd); best = (np.inf, None, 0)
    Xtr, Xva = torch.as_tensor(X[tr]), torch.as_tensor(X[va])
    for ep in range(epochs):
        m.train(); loss = _ce(m(Xtr), Y[tr]); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = _ce(m(Xva), Y[va]).item()
        if vl < best[0] - 1e-4: best = (vl, {k: v.detach().clone() for k, v in m.state_dict().items()}, ep)
        elif ep - best[2] > patience: break
    m.load_state_dict(best[1]); m.eval()
    return m


@torch.no_grad()
def predict_proba(models, X):
    X = torch.as_tensor(X); return sum(torch.softmax(m(X), -1).numpy() for m in models) / len(models)


def fit_predict(kind, P_oof, R_tr, Y_tr, grp_tr, P_te, R_te, seeds=(0, 1, 2)):
    Xtr = tokens(P_oof, R_tr); Xte = tokens(P_te, R_te)
    if kind == 'transformer': make = lambda s: ItemTransformer(seed=s)
    else:
        A = item_graph(Y_tr); make = lambda s: ItemGCN(A, seed=s)
    return predict_proba([_train_one(make, Xtr, Y_tr, grp_tr, s) for s in seeds], Xte)
