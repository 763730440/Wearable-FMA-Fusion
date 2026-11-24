"""Extra autograd ops for the attention and graph fusion baselines (numpy; no deep-learning framework available)."""
import sys; sys.path.insert(0, 'work')
import numpy as np
from nn import T, Adam, concat

def transpose(a):
    out = T(np.swapaxes(a.v, -1, -2), (a,)); out.back = lambda g: a._acc(np.swapaxes(g, -1, -2)); return out

def softmax(a):
    z = a.v - a.v.max(-1, keepdims=True); e = np.exp(z); s = e / e.sum(-1, keepdims=True); out = T(s, (a,))
    out.back = lambda g: a._acc(s * (g - (g * s).sum(-1, keepdims=True))); return out

def layernorm(a, gamma, beta, eps=1e-5):
    mu = a.v.mean(-1, keepdims=True); var = a.v.var(-1, keepdims=True); inv = 1 / np.sqrt(var + eps); xh = (a.v - mu) * inv
    out = T(xh * gamma.v + beta.v, (a, gamma, beta)); D = a.v.shape[-1]
    def back(g):
        if gamma.req: gamma._acc((g * xh).reshape(-1, D).sum(0))
        if beta.req: beta._acc(g.reshape(-1, D).sum(0))
        if a.req:
            gx = g * gamma.v
            a._acc(inv / D * (D * gx - gx.sum(-1, keepdims=True) - xh * (gx * xh).sum(-1, keepdims=True)))
    out.back = back; return out

def cross_entropy(logits, y, mask=None):
    """logits T (N, J, 3); y int (N, J); mean over unmasked entries."""
    z = logits.v - logits.v.max(-1, keepdims=True); lse = np.log(np.exp(z).sum(-1, keepdims=True)); lp = z - lse
    oh = np.eye(logits.v.shape[-1], dtype=np.float32)[y]
    w = np.ones(y.shape, np.float32) if mask is None else mask.astype(np.float32); n = w.sum()
    loss = -(oh * lp).sum(-1); out = T((loss * w).sum() / n, (logits,))
    out.back = lambda g: logits._acc(g * (np.exp(lp) - oh) * w[..., None] / n); return out

def param(*shape, scale=None, rng=None):
    scale = scale if scale is not None else 1 / np.sqrt(shape[0])
    return T(rng.normal(0, scale, shape), req=True)
