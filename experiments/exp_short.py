"""Shortened assessment: estimate the FMA27 total from k recorded items only.
Strategies (latent fusion): adaptive per-patient selection (next item = largest expected reduction of the posterior
variance of the total score, planning as if the item score were revealed; the update then uses the real classifier
evidence), a fixed short form (greedy forward selection on training out-of-fold evidence), random order.
Baselines on the same fixed short form: ridge regression of the total on the recorded items' evidence (trained on
training out-of-fold evidence), and recorded items scored by RF + training mode for the unrecorded items.
Items not administered are unobserved by design (ignorable); evidence of administered items is that item's classifier output."""
import pickle, sys, os, json, numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import RidgeCV
from common import *
from irt import fit_tau, evidence
from mirt import MGPCM

C = pickle.load(open('s1_rom.pkl', 'rb')); r = RIDX; Y = YALL[r]; R0 = M[r].astype(bool); J = len(ITEMS)
KS = (3, 5, 7, 9, 12, 15, 18, 21, 27)

def model(split):
    fn = f'fs_model_{split}.pkl'
    if os.path.exists(fn): return pickle.load(open(fn, 'rb'))
    sp = C[split]; tr = sp['tr']; g = MGPCM(smooth=1.0).fit(Y[tr]); to, tc = fit_tau(g, sp['oof'], Y[tr], mask=R0[tr])
    pickle.dump((g, to, tc), open(fn, 'wb')); return g, to, tc

def predict_subset(g, to, tc, P, mask):
    """mask: (n,J) administered & recorded. Returns item argmax (n,J), total distribution (n,55)."""
    lo = evidence(P, to, mask); lc = evidence(P, tc, mask)
    A = logsumexp(g.LP[None] + lc[:, :, None, :], axis=3)                     # unobserved items: A = 0
    lpost = np.log(g.w + 1e-300)[None] + A.sum(1)
    loo = lpost[:, None, :] - A; loo -= logsumexp(loo, axis=2, keepdims=True)
    it = lo + logsumexp(loo[..., None] + g.LP[None], axis=2)                   # unobserved: lo flat -> predictive
    item = it.argmax(2)
    post = np.exp(lpost - logsumexp(lpost, axis=1, keepdims=True)); tots = []
    for n in range(len(P)):
        c = g.LP + lo[n][:, None, :]; c = np.exp(c - logsumexp(c, axis=2, keepdims=True))
        d = np.zeros((g.LP.shape[1], 2 * J + 1)); d[:, 0] = 1
        for j in range(J):
            nd = d * c[j, :, :1]; nd[:, 1:] += d[:, :-1] * c[j, :, 1:2]; nd[:, 2:] += d[:, :-2] * c[j, :, 2:3]; d = nd
        tots.append(post[n] @ d)
    return item, np.array(tots)

def adaptive_orders(g, tc, P, Rt, to):
    """Per-patient sequential item order (only items with a recording can be administered)."""
    pk = np.exp(g.LP); k3 = np.arange(3.)
    orders = []
    for n in range(len(P)):
        lw = np.log(g.w + 1e-300); obs = np.zeros(J, bool); order = []
        lo = evidence(P[n:n + 1], to)[0]; lc = evidence(P[n:n + 1], tc)[0]
        for step in range(J):
            w = np.exp(lw - logsumexp(lw))
            cm = np.where(obs[:, None, None], np.exp((g.LP + lo[:, None, :]) - logsumexp(g.LP + lo[:, None, :], axis=2, keepdims=True)), pk)   # (J,Q,3)
            m = cm @ k3; v = cm @ k3 ** 2 - m ** 2; Mt = m.sum(0); Vt = v.sum(0)
            best, bj = np.inf, -1
            for j in np.flatnonzero(~obs & Rt[n]):
                ev = 0
                for k in range(3):
                    wk = w * pk[j, :, k]; p = wk.sum()
                    if p < 1e-12: continue
                    wk /= p; Mk = Mt - m[j] + k; Vk = Vt - v[j]
                    ev += p * (wk @ (Vk + Mk ** 2) - (wk @ Mk) ** 2)
                if ev < best: best, bj = ev, j
            if bj < 0: break
            obs[bj] = True; order.append(bj); lw = lw + logsumexp(g.LP[bj] + lc[bj][None], axis=1)
        orders.append(order)
    return orders

def greedy_form(g, to, tc, oof, ytr, mtr):
    chosen = []; tt = ytr.sum(1)
    for _ in range(J):
        best = None
        for j in range(J):
            if j in chosen: continue
            S = np.zeros(J, bool); S[chosen + [j]] = True
            item, _ = predict_subset_fast(g, to, tc, oof, mtr & S[None]); e = np.abs(item.sum(1) - tt).mean()
            if best is None or e < best[0]: best = (e, j)
        chosen.append(best[1])
    return chosen

def predict_subset_fast(g, to, tc, P, mask):
    lo = evidence(P, to, mask); lc = evidence(P, tc, mask)
    A = logsumexp(g.LP[None] + lc[:, :, None, :], axis=3); lpost = np.log(g.w + 1e-300)[None] + A.sum(1)
    loo = lpost[:, None, :] - A; loo -= logsumexp(loo, axis=2, keepdims=True)
    return (lo + logsumexp(loo[..., None] + g.LP[None], axis=2)).argmax(2), None

def feats(P, S): return np.c_[(P[:, S] @ np.arange(3.)), P[:, S].reshape(len(P), -1)]

def run(split):
    sp = C[split]; tr, te, P, oof = sp['tr'], sp['te'], sp['Pte'], sp['oof']; Rt = R0[te]; yt = Y[te]; tt = yt.sum(1)
    g, to, tc = model(split)
    form = greedy_form(g, to, tc, oof, Y[tr], R0[tr]); mode = np.array([np.bincount(Y[tr, j], minlength=3).argmax() for j in range(J)])
    orders = adaptive_orders(g, tc, P, Rt, to)
    out = dict(te=te, form=form, orders=orders, res={})
    for k in KS:
        rr = {}
        S = np.zeros(J, bool); S[form[:k]] = True
        it, d = predict_subset(g, to, tc, P, Rt & S[None]); rr['Latent fusion, fixed short form'] = (it.sum(1), d)
        Ma = np.zeros_like(Rt)
        for n, o in enumerate(orders): Ma[n, o[:k]] = True
        it, d = predict_subset(g, to, tc, P, Ma); rr['Latent fusion, adaptive'] = (it.sum(1), d)
        tots = []
        for seed in range(5):
            rng = np.random.default_rng(seed); Mr = np.zeros_like(Rt)
            for n in range(len(te)): Mr[n, rng.permutation(J)[:k]] = True
            tots.append(predict_subset_fast(g, to, tc, P, Rt & Mr)[0].sum(1))
        rr['Latent fusion, random items'] = (np.array(tots), None)
        Sl = np.flatnonzero(S)
        rg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(feats(oof, Sl), Y[tr].sum(1))
        rr['Ridge regression, fixed short form'] = (np.clip(np.round(rg.predict(feats(P, Sl))), 0, 54), None)
        rr['RF + mode imputation, fixed short form'] = (np.where(S[None], P.argmax(2), mode[None]).sum(1), None)
        out['res'][k] = rr
    pickle.dump(out, open(f'short_{split}.pkl', 'wb')); print(split, 'done', form[:9], flush=True)

if __name__ == '__main__':
    for s in sys.argv[1:]: run(s)
