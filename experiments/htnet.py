"""Hierarchical temporal network (HT-Net) for FMA-UE scoring.
Signal (time series) -> item embedding -> assessment-level context -> item ordinal score,
with supervision at item, subscale and total levels plus a sum-consistency constraint."""
import sys, numpy as np
sys.path.insert(0, 'work')
from nn import T, conv1d, concat, Adam
ARM_IDX = list(range(15)) + [26]; HAND_IDX = list(range(15, 26))
DOMAINS = {'A_shoulder_elbow': list(range(15)), 'C_wrist': list(range(15, 19)), 'D_hand': list(range(19, 26)), 'E_coord': [26]}

def P(rng, *s, scale=None):
    fan = s[0]; return T(rng.normal(0, scale or np.sqrt(2 / fan), size=s), req=True)

class HTNet:
    def __init__(self, seed=0, hier=True, ctx=True, width=32):
        r = np.random.default_rng(seed); w = width; self.hier = hier; self.ctx = ctx
        self.enc = {}
        for name, C in (('arm', 19), ('hand', 5)):
            self.enc[name] = [(P(r, 5 * C, w), T(np.zeros(w), req=True), 1), (P(r, 5 * w, w), T(np.zeros(w), req=True), 2), (P(r, 5 * w, w), T(np.zeros(w), req=True), 4)]
        self.emb = P(r, 27, 2 * w, scale=0.1)
        self.W1 = P(r, 2 * w, w); self.b1 = T(np.zeros(w), req=True)
        self.Wc = P(r, w, w); self.bc = T(np.zeros(w), req=True)
        self.W2 = P(r, 2 * w, w); self.b2 = T(np.zeros(w), req=True)
        self.ws = P(r, w, 1, scale=0.1); self.ib = T(np.zeros(27), req=True)
        self.th1 = T(np.full(27, -0.5), req=True); self.dth = T(np.full(27, 0.5), req=True)
    def params(self):
        ps = [p for L in self.enc.values() for (W, b, _) in L for p in (W, b)]
        return ps + [self.emb, self.W1, self.b1, self.Wc, self.bc, self.W2, self.b2, self.ws, self.ib, self.th1, self.dth]
    def encode(self, x, name):
        h = x
        for W, b, d in self.enc[name]: h = conv1d(h, W, b, 5, d).relu()
        return concat([h.mean(1), h.max(1)], -1)
    def forward(self, arm, hand, mask):
        B = arm.shape[0]
        za = self.encode(T(arm[:, ARM_IDX].reshape(-1, arm.shape[2], 19)), 'arm').reshape(B, 16, -1)
        zh = self.encode(T(hand[:, HAND_IDX].reshape(-1, hand.shape[2], 5)), 'hand').reshape(B, 11, -1)
        z = concat([za[:, :15], zh, za[:, 15:]], 1)  # back to item order (B,27,2w)
        h = ((z + self.emb) @ self.W1 + self.b1).relu()
        m = T(mask[..., None].astype(np.float32))
        if self.ctx:
            c = ((h * m).sum(1) * T(1.0 / np.maximum(mask.sum(1, keepdims=True), 1)))
            c = (c @ self.Wc + self.bc).relu()
            cB = c.reshape(B, 1, -1) + T(np.zeros((B, 27, c.shape[-1])))
            h = (concat([h, cB], -1) @ self.W2 + self.b2).relu()
        s = (h @ self.ws).reshape(B, 27) + self.ib
        t1 = self.th1; t2 = self.th1 + self.dth.softplus()
        p1 = (s - t1).sigmoid(); p2 = (s - t2).sigmoid()
        return p1, p2

def bce(p, y, m):
    eps = 1e-6
    lp = _log(p); l1p = _log(T(np.ones_like(p.v)) - p)
    return ((T(y) * lp + T(1 - y) * l1p) * T(m)).sum() * (-1.0 / max(m.sum(), 1))

def _log(p):
    v = np.clip(p.v, 1e-6, 1); out = T(np.log(v), (p,))
    out.back = lambda g: p._acc(g / v); return out

def loss_fn(p1, p2, Y, M, hier, lam_d=1.0, lam_t=1.0):
    Mf = M.astype(np.float32)
    L = bce(p1, (Y > 0).astype(np.float32), Mf) + bce(p2, (Y > 1).astype(np.float32), Mf)
    if hier:
        E = (p1 + p2) * T(Mf)
        Yf = (Y * M).astype(np.float32)
        for d, idx in DOMAINS.items():
            sc = 2.0 * len(idx)
            diff = (E[:, idx].sum(1) - T(Yf[:, idx].sum(1))) * (1.0 / sc)
            L = L + (diff * diff).mean() * lam_d
        diff = (E.sum(1) - T(Yf.sum(1))) * (1.0 / 54)
        L = L + (diff * diff).mean() * lam_t
    return L

def augment(x, rng):
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
        v = X[:, idx][M[:, idx]]  # (n,L,C)
        st[name] = (v.mean((0, 1)), v.std((0, 1)) + 1e-6)
    return st

def apply_std(ARM, HAND, st):
    return ((ARM - st['arm'][0]) / st['arm'][1]).astype(np.float32), ((HAND - st['hand'][0]) / st['hand'][1]).astype(np.float32)

def train(ARM, HAND, Y, M, seed=0, hier=True, ctx=True, epochs=80, bs=16, lr=2e-3, verbose=False):
    rng = np.random.default_rng(seed + 100)
    net = HTNet(seed, hier, ctx); opt = Adam(net.params(), lr=lr, wd=1e-4)
    n = len(Y)
    for ep in range(epochs):
        perm = rng.permutation(n); tot = 0
        if ep == int(epochs * 0.7): opt.lr = lr * 0.3
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            p1, p2 = net.forward(augment(ARM[b], rng), augment(HAND[b], rng), M[b])
            L = loss_fn(p1, p2, Y[b], M[b], hier); L.backward(); opt.step(); tot += float(L.v) * len(b)
        if verbose and ep % 10 == 0: print(ep, round(tot / n, 4), flush=True)
    return net

def predict(net, ARM, HAND, M, bs=32):
    P1, P2 = [], []
    for i in range(0, len(ARM), bs):
        p1, p2 = net.forward(ARM[i:i + bs], HAND[i:i + bs], M[i:i + bs]); P1.append(p1.v); P2.append(p2.v)
    return np.concatenate(P1), np.concatenate(P2)
