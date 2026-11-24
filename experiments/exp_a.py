"""Direction A feasibility (development cohort only, patient-grouped 5-fold CV).
A0  RF on kinematic ROM features (current primary model)
A1  RF on ROM + Grassmann geometry (projection distance to class-prototype subspaces + eigen-spectrum)
A2  A0 + cross-item context fusion (stage-2 per item on out-of-fold stage-1 probabilities of all items)
A3  A1 + cross-item context fusion
"""
import pickle, json, sys
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

G = pickle.load(open('subspaces.pkl', 'rb'))
BLOCK = {it: ('arm' if it <= 17 or it == 31 else 'hand') for it in ITEMS}
PDIM = {'arm': 3, 'hand': 2}
r = RIDX; Y = YALL[r]; grp = GRP

def geo_feats(tr_rows, rows):
    """tr_rows/rows: absolute indices into D. Prototypes fitted on tr_rows only."""
    out = np.full((len(rows), len(ITEMS), 6), np.nan)
    for j, it in enumerate(ITEMS):
        U = G['UA'] if BLOCK[it] == 'arm' else G['UH']; E = G['EA'] if BLOCK[it] == 'arm' else G['EH']; p = PDIM[BLOCK[it]]
        protos = []
        for c in (0, 1, 2):
            b = [U[i, j] for i in tr_rows if YALL[i, j] == c]
            protos.append(proj_mean(b, p) if len(b) >= 3 else None)
        for a, i in enumerate(rows):
            d = [pdist(U[i, j], P) for P in protos]
            out[a, j, :3] = d
            out[a, j, 3:] = E[i, j]
    return out

def rf():
    return make_pipeline(SimpleImputer(strategy='median'), RandomForestClassifier(300, min_samples_leaf=3, class_weight='balanced', random_state=0, n_jobs=-1))

def stage1(tr_rows, te_rows, use_geo):
    """Returns class-probabilities (n_te, 27, 3)."""
    Xr = D.iloc[tr_rows]; Xe = D.iloc[te_rows]
    if use_geo: gtr = geo_feats(tr_rows, tr_rows); gte = geo_feats(tr_rows, te_rows)
    P = np.zeros((len(te_rows), len(ITEMS), 3))
    for j, it in enumerate(ITEMS):
        Xa = Xr[FC[it]].to_numpy(float); Xb = Xe[FC[it]].to_numpy(float)
        if use_geo: Xa = np.c_[Xa, gtr[:, j]]; Xb = np.c_[Xb, gte[:, j]]
        y = YALL[tr_rows, j].copy()
        vc = pd.Series(y).value_counts()
        for c in vc[vc < 2].index: y[y == c] = 1 if c != 1 else (0 if (y == 0).sum() >= (y == 2).sum() else 2)
        if len(np.unique(y)) == 1: P[:, j, y[0]] = 1; continue
        m = rf().fit(Xa, y); pr = m.predict_proba(Xb)
        for k, c in enumerate(m.classes_): P[:, j, c] = pr[:, k]
    return P

def ctx_feats(P):
    """Assessment-level context: expected item scores of all items -> per item features."""
    Ex = P @ np.array([0, 1, 2.])                     # (n, 27)
    arm = [j for j, it in enumerate(ITEMS) if BLOCK[it] == 'arm']; hand = [j for j, it in enumerate(ITEMS) if BLOCK[it] == 'hand']
    tot = Ex.sum(1, keepdims=True); sa = Ex[:, arm].mean(1, keepdims=True); sh = Ex[:, hand].mean(1, keepdims=True)
    feats = []
    for j in range(len(ITEMS)):
        others = (tot[:, 0] - Ex[:, j]) / (len(ITEMS) - 1)
        feats.append(np.c_[P[:, j, :], others, sa, sh])
    return feats

def stage2(Ptr_oof, ytr, Pte):
    Ftr = ctx_feats(Ptr_oof); Fte = ctx_feats(Pte); out = np.zeros(Pte.shape[:2], int)
    for j in range(len(ITEMS)):
        y = ytr[:, j]
        if len(np.unique(y)) == 1: out[:, j] = y[0]; continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(Ftr[j], y)
        out[:, j] = m.predict(Fte[j])
    return out

def run(use_geo, ctx):
    pred = np.zeros_like(Y)
    for k, (tr, te) in enumerate(GroupKFold(5).split(r, groups=grp)):
        P = stage1(r[tr], r[te], use_geo)
        if not ctx: pred[te] = P.argmax(2); continue
        oof = np.zeros((len(tr), len(ITEMS), 3))
        for itr, ite in GroupKFold(4).split(tr, groups=grp[tr]):
            oof[ite] = stage1(r[tr][itr], r[tr][ite], use_geo)
        pred[te] = stage2(oof, Y[tr], P)
        print('  fold', k, flush=True)
    return pred

def metrics(pred, mask=None):
    m = slice(None) if mask is None else mask
    e = pred[m].sum(1) - Y[m].sum(1)
    return dict(acc=float((pred[m] == Y[m]).mean()), mae=float(np.abs(e).mean()), bias=float(e.mean()))

def paired_boot(pa, pb, B=2000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(grp); idx = {u: np.flatnonzero(grp == u) for u in ug}
    ea = np.abs(pa.sum(1) - Y.sum(1)); eb = np.abs(pb.sum(1) - Y.sum(1)); d = []
    for _ in range(B):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]); d.append(eb[s].mean() - ea[s].mean())
    return float(np.mean(eb - ea)), [float(x) for x in np.percentile(d, [2.5, 97.5])]

if __name__ == '__main__':
    res = {}; preds = {}
    for name, g, c in [('A0', False, False), ('A1', True, False), ('A2', False, True), ('A3', True, True)]:
        print(name, flush=True); preds[name] = run(g, c); res[name] = metrics(preds[name]); print(name, res[name], flush=True)
    for name in ('A1', 'A2', 'A3'): res[name]['mae_diff_vs_A0'] = paired_boot(preds['A0'], preds[name])
    sev = Y.sum(1) < 40
    for name in preds: res[name]['instr<40'] = metrics(preds[name], sev); res[name]['instr>=40'] = metrics(preds[name], ~sev)
    json.dump(res, open('res_a.json', 'w'), indent=1); np.savez('pred_a.npz', **preds)
    print(json.dumps(res, indent=1))
