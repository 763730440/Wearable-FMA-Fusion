"""Hierarchical temporal network (HT-Net) for FMA-UE scoring, in PyTorch.
Signal (time series) -> item embedding -> assessment-level context -> item ordinal score,
with supervision at item, subscale and total levels plus a sum-consistency constraint.
hier=False, ctx=False gives the flat TCN baseline (items scored independently)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARM_IDX = list(range(15)) + [26]; HAND_IDX = list(range(15, 26))
DOMAINS = {'A_shoulder_elbow': list(range(15)), 'C_wrist': list(range(15, 19)), 'D_hand': list(range(19, 26)), 'E_coord': [26]}


def _he(rng, fan_in, fan_out, scale=None):
    """He-normal weights drawn from a NumPy generator, so that a given seed gives the same initial weights on every platform."""
    return nn.Parameter(torch.tensor(rng.normal(0, scale or np.sqrt(2 / fan_in), size=(fan_in, fan_out)), dtype=torch.float32))


class DilatedTCN(nn.Module):
    """Three 1-D convolutions (kernel 5, dilation 1/2/4, same padding, ReLU), then mean and max pooling over time."""
    def __init__(self, c_in, width, rng, k=5, dilations=(1, 2, 4)):
        super().__init__()
        self.k = k; self.dilations = dilations
        self.convs = nn.ModuleList()
        for j, d in enumerate(dilations):
            cin = c_in if j == 0 else width
            conv = nn.Conv1d(cin, width, k, padding=d * (k - 1) // 2, dilation=d)
            W = rng.normal(0, np.sqrt(2 / (k * cin)), size=(k * cin, width))      # fan-in = k * c_in
            with torch.no_grad():
                conv.weight.copy_(torch.tensor(W, dtype=torch.float32).reshape(k, cin, width).permute(2, 1, 0))
                conv.bias.zero_()
            self.convs.append(conv)

    def forward(self, x):                      # x: (N, L, C)
        h = x.transpose(1, 2)
        for conv in self.convs: h = torch.relu(conv(h))
        return torch.cat([h.mean(2), h.amax(2)], -1)


class HTNet(nn.Module):
    def __init__(self, seed=0, hier=True, ctx=True, width=32):
        super().__init__()
        r = np.random.default_rng(seed); w = width; self.hier = hier; self.ctx = ctx
        self.enc = nn.ModuleDict({'arm': DilatedTCN(19, w, r), 'hand': DilatedTCN(5, w, r)})
        self.emb = nn.Parameter(torch.tensor(r.normal(0, 0.1, size=(27, 2 * w)), dtype=torch.float32))
        self.W1 = _he(r, 2 * w, w); self.b1 = nn.Parameter(torch.zeros(w))
        self.Wc = _he(r, w, w); self.bc = nn.Parameter(torch.zeros(w))
        self.W2 = _he(r, 2 * w, w); self.b2 = nn.Parameter(torch.zeros(w))
        self.ws = _he(r, w, 1, scale=0.1); self.ib = nn.Parameter(torch.zeros(27))
        self.th1 = nn.Parameter(torch.full((27,), -0.5)); self.dth = nn.Parameter(torch.full((27,), 0.5))

    def forward(self, arm, hand, mask):
        arm = torch.as_tensor(arm, dtype=torch.float32); hand = torch.as_tensor(hand, dtype=torch.float32); B = arm.shape[0]
        za = self.enc['arm'](arm[:, ARM_IDX].reshape(-1, arm.shape[2], 19)).reshape(B, 16, -1)
        zh = self.enc['hand'](hand[:, HAND_IDX].reshape(-1, hand.shape[2], 5)).reshape(B, 11, -1)
        z = torch.cat([za[:, :15], zh, za[:, 15:]], 1)                  # back to item order (B, 27, 2w)
        h = torch.relu((z + self.emb) @ self.W1 + self.b1)
        if self.ctx:
            m = torch.as_tensor(mask[..., None], dtype=torch.float32)
            c = (h * m).sum(1) / m.sum(1).clamp(min=1)                    # mean over recorded items
            c = torch.relu(c @ self.Wc + self.bc)
            h = torch.relu(torch.cat([h, c[:, None].expand(B, 27, -1)], -1) @ self.W2 + self.b2)
        s = (h @ self.ws).reshape(B, 27) + self.ib
        t1 = self.th1; t2 = self.th1 + F.softplus(self.dth)               # ordered CORAL thresholds
        return torch.sigmoid(s - t1), torch.sigmoid(s - t2)


def _log(p):
    """log(p) clipped at 1e-6; the gradient is 1 / clip(p) and is never zeroed (keeps saturated items trainable)."""
    v = p.clamp(1e-6, 1).detach()
    return torch.log(v) + (p - p.detach()) / v


def bce(p, y, m):
    y = torch.as_tensor(y); m = torch.as_tensor(m)
    return -((y * _log(p) + (1 - y) * _log(1 - p)) * m).sum() / max(float(m.sum()), 1.0)


def loss_fn(p1, p2, Y, M, hier, lam_d=1.0, lam_t=1.0):
    """Item-level CORAL BCE; with hier=True also subscale and total sum-consistency (squared, scaled to the subscale range)."""
    Mf = M.astype(np.float32)
    L = bce(p1, (Y > 0).astype(np.float32), Mf) + bce(p2, (Y > 1).astype(np.float32), Mf)
    if hier:
        E = (p1 + p2) * torch.as_tensor(Mf)
        Yf = torch.as_tensor((Y * M).astype(np.float32))
        for idx in DOMAINS.values():
            diff = (E[:, idx].sum(1) - Yf[:, idx].sum(1)) / (2.0 * len(idx))
            L = L + lam_d * (diff ** 2).mean()
        diff = (E.sum(1) - Yf.sum(1)) / 54
        L = L + lam_t * (diff ** 2).mean()
    return L


def augment(x, rng):
    """Per-channel amplitude scaling, random time crop-and-stretch, Gaussian noise (NumPy generator)."""
    B, I, L, C = x.shape
    x = x * rng.uniform(0.85, 1.15, size=(B, I, 1, C)).astype(np.float32)
    out = np.empty_like(x)
    for b in range(B):
        a0 = rng.integers(0, L // 8 + 1); a1 = L - rng.integers(0, L // 8 + 1)
        u = np.linspace(a0, a1 - 1, L); i0 = np.floor(u).astype(int); fr = (u - i0)[:, None]; i1 = np.minimum(i0 + 1, L - 1)
        out[b] = x[b][:, i0] * (1 - fr) + x[b][:, i1] * fr
    return out + rng.normal(0, 0.02, out.shape).astype(np.float32)


def standardize_fit(ARM, HAND, M):
    st = {}
    for name, X, idx in (('arm', ARM, ARM_IDX), ('hand', HAND, HAND_IDX)):
        v = X[:, idx][M[:, idx]]  # (n, L, C)
        st[name] = (v.mean((0, 1)), v.std((0, 1)) + 1e-6)
    return st


def apply_std(ARM, HAND, st):
    return ((ARM - st['arm'][0]) / st['arm'][1]).astype(np.float32), ((HAND - st['hand'][0]) / st['hand'][1]).astype(np.float32)


def train(ARM, HAND, Y, M, seed=0, hier=True, ctx=True, epochs=80, bs=16, lr=2e-3, verbose=False):
    """Adam (L2 weight decay 1e-4), batch 16, 80 epochs, learning rate x0.3 after 70% of the epochs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 100)                  # augmentation and batch order
    net = HTNet(seed, hier, ctx)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Y); net.train()
    for ep in range(epochs):
        perm = rng.permutation(n); tot = 0.0
        if ep == int(epochs * 0.7):
            for g in opt.param_groups: g['lr'] = lr * 0.3
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            p1, p2 = net(augment(ARM[b], rng), augment(HAND[b], rng), M[b])
            L = loss_fn(p1, p2, Y[b], M[b], hier)
            opt.zero_grad(); L.backward(); opt.step(); tot += L.item() * len(b)
        if verbose and ep % 10 == 0: print(ep, round(tot / n, 4), flush=True)
    return net


@torch.no_grad()
def predict(net, ARM, HAND, M, bs=32):
    """Returns P(score >= 1) and P(score = 2) for every item."""
    net.eval(); P1, P2 = [], []
    for i in range(0, len(ARM), bs):
        p1, p2 = net(ARM[i:i + bs], HAND[i:i + bs], M[i:i + bs]); P1.append(p1.numpy()); P2.append(p2.numpy())
    return np.concatenate(P1), np.concatenate(P2)
