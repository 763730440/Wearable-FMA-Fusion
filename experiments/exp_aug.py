"""Learned fusers trained with missing-data augmentation (reviewer-requested strong baselines).
The Transformer and GCN of exp_learned.py are retrained with recordings removed at random from the training tokens,
with a fresh mask in every epoch. Three augmentation schemes, all using the training labels:
  mcar  - per assessment a rate p ~ U(0, 1) is drawn and every item loses its recording with probability p
  mnar  - per assessment p ~ U(0, 1); items scored 0 lose their recording with probability p, others with 0.02
  mixed - per assessment one of the two schemes is chosen with probability 1/2
The early-stopping split uses one fixed augmented mask drawn with the same scheme.
Evaluation: Xi'an CV, both transfer directions, controlled missingness (same protocol and seeds as exp_sim.py and
exp_learned.py) and the Beijing locked test. Usage: python experiments/exp_aug.py transformer|gcn"""
import pickle, json, sys
import numpy as np, torch
from common import *
import learned_fusion as LF
from eval_all import metrics, boot_diff

torch.set_num_threads(1)
KIND = sys.argv[1]
SCHEMES = ('mcar', 'mnar', 'mixed')
C = pickle.load(open('s1_rom.pkl', 'rb')); r = RIDX; Y = YALL[r]; R = M[r].astype(bool)


def aug_mask(rng, Rb, Yb, scheme):
    n = len(Rb); p = rng.random((n, 1)); u = rng.random(Rb.shape)
    mcar = u < p; mnar = u < np.where(Yb == 0, p, 0.02)
    if scheme == 'mcar': drop = mcar
    elif scheme == 'mnar': drop = mnar
    else: drop = np.where(rng.random((n, 1)) < 0.5, mcar, mnar)
    return Rb & ~drop


def train_aug(make, P, Rb, Yb, grp, seed, scheme, epochs=400, lr=3e-3, wd=1e-3, patience=40):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed); ug = np.unique(grp); val_g = rng.choice(ug, max(1, len(ug) // 5), replace=False)
    va = np.isin(grp, val_g); tr = ~va
    arng = np.random.default_rng(1000 + seed)
    Xva = torch.as_tensor(LF.tokens(P[va], aug_mask(arng, Rb[va], Yb[va], scheme)))
    m = make(seed); opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd); best = (np.inf, None, 0)
    for ep in range(epochs):
        Xtr = torch.as_tensor(LF.tokens(P[tr], aug_mask(arng, Rb[tr], Yb[tr], scheme)))
        m.train(); loss = LF._ce(m(Xtr), Yb[tr]); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = LF._ce(m(Xva), Yb[va]).item()
        if vl < best[0] - 1e-4: best = (vl, {k: v.detach().clone() for k, v in m.state_dict().items()}, ep)
        elif ep - best[2] > patience: break
    m.load_state_dict(best[1]); m.eval()
    return m


def fit(P, Rb, Yb, grp, scheme):
    make = (lambda s: LF.ItemTransformer(seed=s)) if KIND == 'transformer' else (lambda s, A=LF.item_graph(Yb): LF.ItemGCN(A, seed=s))
    return [train_aug(make, P, Rb, Yb, grp, s, scheme) for s in (0, 1, 2)]


res = {}
for scheme in SCHEMES:
    out = {'cv': {}, 'transfer': {}, 'beijing': {}, 'sim': {}}
    fold_models = {}; cvp = np.zeros_like(Y)
    for f in range(5):
        sp = C[f'cv{f}']; tr, te = sp['tr'], sp['te']
        fold_models[f] = fit(sp['oof'], R[tr], Y[tr], GRP[tr], scheme)
        cvp[te] = LF.predict_proba(fold_models[f], LF.tokens(sp['Pte'], R[te])).argmax(2)
    out['cv'] = metrics(cvp, Y)
    pp = np.load('paper_preds.npz'); out['cv']['vs_IRT-2D'] = boot_diff(cvp, pp['cv_IRT-2D'], Y, GRP)
    print(KIND, scheme, 'cv', round(out['cv']['acc'], 4), round(out['cv']['mae'], 3), flush=True)
    for s in ('c1to2', 'c2to1'):
        sp = C[s]; tr, te = sp['tr'], sp['te']
        p = LF.predict_proba(fit(sp['oof'], R[tr], Y[tr], GRP[tr], scheme), LF.tokens(sp['Pte'], R[te])).argmax(2)
        out['transfer'][s] = metrics(p, Y[te])
    print(KIND, scheme, 'transfer', {s: round(v['mae'], 3) for s, v in out['transfer'].items()}, flush=True)
    pla = PIDX; Yp = YALL[pla]; Rp = M[pla].astype(bool); pid = D.iloc[pla].pid.to_numpy(); tp = Yp.sum(1); sevp = tp < 40
    e = np.load('ext_probs_rom.npz'); rfb = e['P'].argmax(2)
    p = LF.predict_proba(fit(e['oof'], R, Y, GRP, scheme), LF.tokens(e['P'], Rp)).argmax(2)
    m = metrics(p, Yp); er = p.sum(1) - tp
    m.update(sev_mae=float(np.abs(er[sevp]).mean()), sev_bias=float(er[sevp].mean()), mild_mae=float(np.abs(er[~sevp]).mean()),
             mild_bias=float(er[~sevp].mean()), vs_RF=boot_diff(rfb, p, Yp, pid))
    out['beijing'] = m
    print(KIND, scheme, 'beijing', round(m['acc'], 4), round(m['mae'], 3), 'sev bias', round(m['sev_bias'], 2), flush=True)
    for mech in ('MNAR', 'MCAR'):
        for prate in (0.0, 0.25, 0.5, 0.75, 1.0):
            acc, mae = [], []
            for seed in range(3):
                rng = np.random.default_rng(seed); pred = np.zeros_like(Y)
                for f in range(5):
                    sp = C[f'cv{f}']; te = sp['te']; yt = Y[te]; u = rng.random(yt.shape)
                    drop = (u < np.where(yt == 0, prate, 0.02)) if mech == 'MNAR' else (u < prate)
                    pred[te] = LF.predict_proba(fold_models[f], LF.tokens(sp['Pte'], R[te] & ~drop)).argmax(2)
                acc.append((pred == Y).mean()); mae.append(np.abs(pred.sum(1) - Y.sum(1)).mean())
            out['sim'][f'{mech}_{prate}'] = dict(acc=float(np.mean(acc)), mae=float(np.mean(mae)))
        print(KIND, scheme, mech, {k: round(v['mae'], 2) for k, v in out['sim'].items() if k.startswith(mech)}, flush=True)
    res[scheme] = out
    json.dump(res, open(f'res_aug_{KIND}.json', 'w'), indent=1)
print('saved')
