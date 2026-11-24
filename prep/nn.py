"""Minimal reverse-mode autograd in numpy (no PyTorch available in this environment)."""
import numpy as np

class T:
    def __init__(self, v, parents=(), back=None, req=False):
        self.v = np.asarray(v, dtype=np.float32); self.g = None
        self.parents = parents; self.back = back; self.req = req or any(p.req for p in parents)
    @property
    def shape(self): return self.v.shape
    def _acc(self, g):
        self.g = g if self.g is None else self.g + g
    def backward(self):
        order, seen = [], set()
        def topo(t):
            if id(t) in seen or not t.req: return
            seen.add(id(t))
            for p in t.parents: topo(p)
            order.append(t)
        topo(self); self.g = np.ones_like(self.v)
        for t in reversed(order):
            if t.back is not None and t.g is not None: t.back(t.g)
    # ---- ops
    def __add__(a, b):
        b = b if isinstance(b, T) else T(b)
        out = T(a.v + b.v, (a, b))
        def back(g):
            if a.req: a._acc(_unb(g, a.v.shape))
            if b.req: b._acc(_unb(g, b.v.shape))
        out.back = back; return out
    def __sub__(a, b): return a + (b * -1.0 if isinstance(b, T) else T(-np.asarray(b)))
    def __mul__(a, b):
        b = b if isinstance(b, T) else T(b)
        out = T(a.v * b.v, (a, b))
        def back(g):
            if a.req: a._acc(_unb(g * b.v, a.v.shape))
            if b.req: b._acc(_unb(g * a.v, b.v.shape))
        out.back = back; return out
    def __matmul__(a, b):
        out = T(a.v @ b.v, (a, b))
        def back(g):
            if a.req: a._acc(g @ np.swapaxes(b.v, -1, -2) if b.v.ndim > 1 else np.outer(g, b.v))
            if b.req:
                ga = np.swapaxes(a.v, -1, -2) @ g
                b._acc(_unb(ga, b.v.shape) if ga.shape != b.v.shape else ga)
        out.back = back; return out
    def relu(a):
        m = a.v > 0; out = T(a.v * m, (a,))
        out.back = lambda g: a._acc(g * m); return out
    def sigmoid(a):
        s = 1 / (1 + np.exp(-np.clip(a.v, -30, 30))); out = T(s, (a,))
        out.back = lambda g: a._acc(g * s * (1 - s)); return out
    def softplus(a):
        out = T(np.logaddexp(0, a.v), (a,)); s = 1 / (1 + np.exp(-np.clip(a.v, -30, 30)))
        out.back = lambda g: a._acc(g * s); return out
    def sum(a, axis=None, keepdims=False):
        out = T(a.v.sum(axis=axis, keepdims=keepdims), (a,))
        def back(g):
            if axis is not None and not keepdims: g = np.expand_dims(g, axis)
            a._acc(np.broadcast_to(g, a.v.shape).copy())
        out.back = back; return out
    def mean(a, axis=None, keepdims=False):
        n = a.v.size if axis is None else a.v.shape[axis]
        return a.sum(axis, keepdims) * (1.0 / n)
    def max(a, axis):
        idx = a.v.argmax(axis); out = T(np.take_along_axis(a.v, np.expand_dims(idx, axis), axis).squeeze(axis), (a,))
        def back(g):
            z = np.zeros_like(a.v); np.put_along_axis(z, np.expand_dims(idx, axis), np.expand_dims(g, axis), axis); a._acc(z)
        out.back = back; return out
    def reshape(a, *s):
        out = T(a.v.reshape(*s), (a,)); out.back = lambda g: a._acc(g.reshape(a.v.shape)); return out
    def __getitem__(a, k):
        out = T(a.v[k], (a,))
        def back(g):
            z = np.zeros_like(a.v); np.add.at(z, k, g); a._acc(z)
        out.back = back; return out

def _unb(g, shape):
    while g.ndim > len(shape): g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1: g = g.sum(i, keepdims=True)
    return g

def concat(ts, axis=-1):
    out = T(np.concatenate([t.v for t in ts], axis), tuple(ts))
    sizes = np.cumsum([0] + [t.v.shape[axis] for t in ts])
    def back(g):
        for t, s0, s1 in zip(ts, sizes[:-1], sizes[1:]):
            if t.req: t._acc(np.take(g, np.arange(s0, s1), axis=axis))
    out.back = back; return out

def conv1d(x, W, b, k, dil):
    """x: (N,T,C) T; W: (k*C, Co); same padding."""
    N, L, C = x.v.shape; pad = dil * (k - 1) // 2
    idx = np.arange(L)[:, None] + (np.arange(k) * dil)[None, :]  # (L,k) into padded
    xp = np.pad(x.v, ((0, 0), (pad, pad), (0, 0)))
    P = xp[:, idx, :].reshape(N, L, k * C)
    out = T(P @ W.v + b.v, (x, W, b))
    def back(g):
        if W.req: W._acc(P.reshape(-1, k * C).T @ g.reshape(-1, g.shape[-1]))
        if b.req: b._acc(g.reshape(-1, g.shape[-1]).sum(0))
        if x.req:
            gP = (g @ W.v.T).reshape(N, L, k, C); gx = np.zeros_like(xp)
            for j in range(k): gx[:, idx[:, j], :] += gP[:, :, j, :]
            x._acc(gx[:, pad:pad + L, :])
    out.back = back; return out

class Adam:
    def __init__(self, params, lr=1e-3, wd=1e-4, b1=.9, b2=.999):
        self.p = params; self.lr = lr; self.wd = wd; self.b1 = b1; self.b2 = b2; self.t = 0
        self.m = [np.zeros_like(p.v) for p in params]; self.s = [np.zeros_like(p.v) for p in params]
    def step(self):
        self.t += 1
        for i, p in enumerate(self.p):
            if p.g is None: continue
            g = p.g + self.wd * p.v
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g; self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t); sh = self.s[i] / (1 - self.b2 ** self.t)
            p.v -= self.lr * mh / (np.sqrt(sh) + 1e-8); p.g = None
