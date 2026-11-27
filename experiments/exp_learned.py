"""Attention and graph fusion baselines on the same item evidence: Xi'an CV, center transfer, Beijing, controlled missingness."""
import pickle, json, sys
from common import *
from learned_fusion import fit_predict
import learned_fusion as LF
from eval_all import metrics, boot_diff

C = pickle.load(open('s1_rom.pkl', 'rb')); r = RIDX; Y = YALL[r]; R = M[r].astype(bool)
KINDS = ('transformer', 'gcn')
res = {'cv': {}, 'transfer': {}, 'beijing': {}, 'sim': {}}
models_cache = {}
# keep trained models for the simulation: refit is cheap enough, so fit_predict is called per scenario on the same seeds
cvp = {k: np.zeros_like(Y) for k in KINDS}; cvprob = {k: np.zeros(Y.shape + (3,)) for k in KINDS}
for f in range(5):
    sp = C[f'cv{f}']; tr, te = sp['tr'], sp['te']
    for k in KINDS:
        pr = fit_predict(k, sp['oof'], R[tr], Y[tr], GRP[tr], sp['Pte'], R[te]); cvprob[k][te] = pr; cvp[k][te] = pr.argmax(2)
    print('cv fold', f, flush=True)
rfcv = np.zeros_like(Y)
for f in range(5): rfcv[C[f'cv{f}']['te']] = C[f'cv{f}']['Pte'].argmax(2)
for k in KINDS:
    res['cv'][k] = metrics(cvp[k], Y); res['cv'][k]['vs_RF'] = boot_diff(rfcv, cvp[k], Y, GRP)
    t = Y.sum(1); sev = t < 40; e = cvp[k].sum(1) - t
    res['cv'][k].update(sev_mae=float(np.abs(e[sev]).mean()), mild_mae=float(np.abs(e[~sev]).mean()), sev_bias=float(e[sev].mean()), mild_bias=float(e[~sev].mean()))
pp = np.load('paper_preds.npz')
for k in KINDS:
    res['cv'][k]['vs_IRT-2D'] = boot_diff(cvp[k], pp['cv_IRT-2D'], Y, GRP)   # negative = latent fusion better
print('cv', {k: (round(v['acc'], 4), round(v['mae'], 3)) for k, v in res['cv'].items()}, flush=True)
for s in ('c1to2', 'c2to1'):
    sp = C[s]; tr, te = sp['tr'], sp['te']; res['transfer'][s] = {}
    for k in KINDS:
        p = fit_predict(k, sp['oof'], R[tr], Y[tr], GRP[tr], sp['Pte'], R[te]).argmax(2); m = metrics(p, Y[te])
        tt = Y[te].sum(1); mild = tt >= 40; m['mild_mae'] = float(np.abs(p.sum(1) - tt)[mild].mean()); res['transfer'][s][k] = m
    print(s, {k: (round(v['acc'], 4), round(v['mae'], 3)) for k, v in res['transfer'][s].items()}, flush=True)
# Beijing (skipped when the private Beijing cohort is not available)
if HAS_BEIJING:
    pla = PIDX; Yp = YALL[pla]; Rp = M[pla].astype(bool); pid = D.iloc[pla].pid.to_numpy(); tp = Yp.sum(1); sevp = tp < 40
    e = np.load('ext_probs_rom.npz'); rfb = e['P'].argmax(2)
    for k in KINDS:
        p = fit_predict(k, e['oof'], R, Y, GRP, e['P'], Rp).argmax(2)
        for name, q in ((k, p), (k + ' + zero rule', np.where(Rp, p, 0))):
            m = metrics(q, Yp); er = q.sum(1) - tp
            m.update(sev_mae=float(np.abs(er[sevp]).mean()), sev_bias=float(er[sevp].mean()), mild_mae=float(np.abs(er[~sevp]).mean()), mild_bias=float(er[~sevp].mean()),
                     vs_RF=boot_diff(rfb, q, Yp, pid))
            res['beijing'][name] = m
    print('beijing', {k: (round(v['acc'], 4), round(v['mae'], 3)) for k, v in res['beijing'].items()}, flush=True)
# controlled missingness (same protocol and seeds as exp_sim.py); models refitted per fold once, inputs perturbed
fold_models = {}
for f in range(5):
    sp = C[f'cv{f}']; tr = sp['tr']; Xtr = LF.tokens(sp['oof'], R[tr]); fold_models[f] = {}
    for k in KINDS:
        make = (lambda s: LF.ItemTransformer(seed=s)) if k == 'transformer' else (lambda s, A=LF.item_graph(Y[tr]): LF.ItemGCN(A, seed=s))
        fold_models[f][k] = [LF._train_one(make, Xtr, Y[tr], GRP[tr], s) for s in (0, 1, 2)]
for mech in ('MNAR', 'MCAR'):
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        acc = {k: [] for k in KINDS}; mae = {k: [] for k in KINDS}
        for seed in range(3):
            rng = np.random.default_rng(seed); preds = {k: np.zeros_like(Y) for k in KINDS}
            for f in range(5):
                sp = C[f'cv{f}']; te = sp['te']; yt = Y[te]; u = rng.random(yt.shape)
                drop = (u < np.where(yt == 0, p, 0.02)) if mech == 'MNAR' else (u < p); Rt = R[te] & ~drop
                X = LF.tokens(sp['Pte'], Rt)
                for k in KINDS:
                    preds[k][te] = LF.predict_proba(fold_models[f][k], X).argmax(2)
            for k in KINDS:
                acc[k].append((preds[k] == Y).mean()); mae[k].append(np.abs(preds[k].sum(1) - Y.sum(1)).mean())
        res['sim'][f'{mech}_{p}'] = {k: dict(acc=float(np.mean(acc[k])), mae=float(np.mean(mae[k]))) for k in KINDS}
        print(mech, p, res['sim'][f'{mech}_{p}'], flush=True)
np.savez('learned_preds.npz', **{f'cv_{k}': v for k, v in cvp.items()})
json.dump(res, open('results_learned.json', 'w'), indent=1); print('saved')
