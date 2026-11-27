"""Deployment-condition analyses (reviewer-requested).
(a) MALT batch size: the missingness rates are re-estimated by EM from batches of n unlabeled assessments
    (n = 1, 5, 10, 20, whole test set) instead of all assessments of the site; controlled missingness (same masks
    as exp_sim.py) and Beijing (20 random partitions into batches).
(b) Learned fusers with unlabeled adaptation on the same batch as MALT (the whole test fold / all Beijing assessments):
    - TENT: test-time entropy minimisation of the normalisation parameters of the mixed-augmentation fuser
      (Wang et al., ICLR 2021), 10 Adam steps, lr 1e-3;
    - entropy selection: per batch, the MCAR- or MNAR-augmented fuser whose predictions on the unlabeled batch have
      the lower mean entropy.
Usage: python experiments/exp_batch.py malt | transformer | gcn"""
import pickle, json, sys, copy
import numpy as np
from common import *

WHAT = sys.argv[1]
C = pickle.load(open('s1_rom.pkl', 'rb')); r = RIDX; Y = YALL[r]; R0 = M[r].astype(bool)
CONDS = (('MNAR', 0.5), ('MNAR', 1.0), ('MCAR', 0.5))


def masks():
    """Yield (mech, p, seed, fold, Rt) with exactly the random draws of exp_sim.py / exp_learned.py."""
    for mech, p in CONDS:
        for seed in range(3):
            rng = np.random.default_rng(seed)
            for f in range(5):
                sp = C[f'cv{f}']; te = sp['te']; yt = Y[te]; u = rng.random(yt.shape)
                drop = (u < np.where(yt == 0, p, 0.02)) if mech == 'MNAR' else (u < p)
                yield mech, p, seed, f, R0[te] & ~drop


pla = PIDX; Yp = YALL[pla]; Rp = M[pla].astype(bool); tp = Yp.sum(1)
res = {}
if WHAT == 'malt':
    from irt import fit_tau
    from mirt import MGPCM
    from missirt import fuse_m, rho_from_labels, estimate_rho
    models = {}
    for f in range(5):
        sp = C[f'cv{f}']; g = MGPCM(smooth=1.0).fit(Y[sp['tr']]); to, tc = fit_tau(g, sp['oof'], Y[sp['tr']], mask=R0[sp['tr']])
        models[f] = (g, to, tc, rho_from_labels(Y[sp['tr']], R0[sp['tr']]))
    SIZES = (1, 5, 10, 20, 'all')
    acc = {}
    for mech, p, seed, f, Rt in masks():
        sp = C[f'cv{f}']; te = sp['te']; Pte = sp['Pte']; g, to, tc, rho0 = models[f]
        brng = np.random.default_rng(100 + seed)
        for n in SIZES:
            order = brng.permutation(len(te)); pred = np.zeros((len(te), Y.shape[1]), int)
            batches = [order] if n == 'all' else [order[i:i + n] for i in range(0, len(te), n)]
            for b in batches:
                rho = estimate_rho(g, Pte[b], Rt[b], rho0, to, tc, iters=15)
                pred[b] = fuse_m(g, Pte[b], Rt[b], rho, to, tc).argmax(2)
            key = f'{mech}_{p}_n{n}'
            acc.setdefault(key, {}).setdefault(seed, np.zeros_like(Y))[te] = pred
    for key, d in acc.items():
        res[key] = float(np.mean([np.abs(pp.sum(1) - Y.sum(1)).mean() for pp in d.values()]))
        print(key, round(res[key], 3), flush=True)
    # Beijing
    if HAS_BEIJING:   # Beijing (private cohort)
        e = np.load('ext_probs_rom.npz'); P, oof = e['P'], e['oof']; MR = R0
        g = MGPCM(smooth=1.0).fit(Y); to, tc = fit_tau(g, oof, Y, mask=MR); rho0 = rho_from_labels(Y, MR)
        for n in SIZES:
            maes, sev = [], []
            for rep in range(1 if n == 'all' else 20):
                order = np.random.default_rng(rep).permutation(len(pla)); pred = np.zeros_like(Yp)
                batches = [order] if n == 'all' else [order[i:i + n] for i in range(0, len(pla), n)]
                for b in batches:
                    rho = estimate_rho(g, P[b], Rp[b], rho0, to, tc)
                    pred[b] = fuse_m(g, P[b], Rp[b], rho, to, tc).argmax(2)
                er = pred.sum(1) - tp; maes.append(np.abs(er).mean()); sev.append(er[tp < 40].mean())
            res[f'beijing_n{n}'] = dict(mae=float(np.mean(maes)), sev_bias=float(np.mean(sev)))
            print('beijing', n, res[f'beijing_n{n}'], flush=True)
else:
    import torch
    import learned_fusion as LF
    from exp_aug import train_aug, KIND  # noqa: E402  (exp_aug reads sys.argv[1] as the kind)
    torch.set_num_threads(1)

    def fit(P, Rb, Yb, grp, scheme):
        make = (lambda s: LF.ItemTransformer(seed=s)) if KIND == 'transformer' else (lambda s, A=LF.item_graph(Yb): LF.ItemGCN(A, seed=s))
        return [train_aug(make, P, Rb, Yb, grp, s, scheme) for s in (0, 1, 2)]

    def norm_params(m):
        return [p for n, p in m.named_parameters() if n.split('.')[-1] in ('g1', 'b1', 'g2', 'b2', 'gf', 'bf') or n.startswith(('g.', 'be.'))]

    def tent(models, X, steps=10, lr=1e-3):
        out = []
        Xt = torch.as_tensor(X)
        for m0 in models:
            m = copy.deepcopy(m0); ps = norm_params(m)
            for p in m.parameters(): p.requires_grad_(False)
            for p in ps: p.requires_grad_(True)
            opt = torch.optim.Adam(ps, lr=lr)
            for _ in range(steps):
                pr = torch.softmax(m(Xt), -1); loss = -(pr * torch.log(pr + 1e-8)).sum(-1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            m.eval(); out.append(m)
        return LF.predict_proba(out, X)

    def ent(pr): return float(-(pr * np.log(pr + 1e-8)).sum(-1).mean())

    fm = {f: {s: fit(C[f'cv{f}']['oof'], R0[C[f'cv{f}']['tr']], Y[C[f'cv{f}']['tr']], GRP[C[f'cv{f}']['tr']], s)
              for s in ('mcar', 'mnar', 'mixed')} for f in range(5)}
    print('fold models ready', flush=True)
    pred = {}
    for mech, p, seed, f, Rt in masks():
        te = C[f'cv{f}']['te']; X = LF.tokens(C[f'cv{f}']['Pte'], Rt)
        pa = LF.predict_proba(fm[f]['mcar'], X); pb = LF.predict_proba(fm[f]['mnar'], X)
        sel = pa if ent(pa) < ent(pb) else pb
        for name, pr in (('tent_mixed', tent(fm[f]['mixed'], X)), ('entropy_select', sel)):
            pred.setdefault(f'{mech}_{p}_{name}', {}).setdefault(seed, np.zeros_like(Y))[te] = pr.argmax(2)
    for key, d in pred.items():
        res[key] = float(np.mean([np.abs(pp.sum(1) - Y.sum(1)).mean() for pp in d.values()]))
        print(key, round(res[key], 3), flush=True)
    if HAS_BEIJING:   # Beijing (private cohort)
        e = np.load('ext_probs_rom.npz'); X = LF.tokens(e['P'], Rp)
        bm = {s: fit(e['oof'], R0, Y, GRP, s) for s in ('mcar', 'mnar', 'mixed')}
        pa = LF.predict_proba(bm['mcar'], X); pb = LF.predict_proba(bm['mnar'], X)
        for name, pr in (('tent_mixed', tent(bm['mixed'], X)), ('entropy_select', pa if ent(pa) < ent(pb) else pb)):
            er = pr.argmax(2).sum(1) - tp
            res[f'beijing_{name}'] = dict(mae=float(np.abs(er).mean()), sev_bias=float(er[tp < 40].mean()), chose=('mcar' if ent(pa) < ent(pb) else 'mnar') if name == 'entropy_select' else None)
            print('beijing', name, res[f'beijing_{name}'], flush=True)
json.dump(res, open(f'res_batch_{WHAT}.json', 'w'), indent=1); print('saved')
