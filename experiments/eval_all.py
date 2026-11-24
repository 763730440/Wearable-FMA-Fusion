"""All results for the fusion manuscript. Xi'an: pooled patient-grouped 5-fold CV and center transfer.
Beijing: single locked test (configurations fixed in LOCK_FINAL.txt / LOCK_MA.txt). Outputs results_paper.json."""
import pickle, json
from common import *
from exp_a import stage2
from irt import GPCM, fuse, fit_tau
from mirt import MGPCM
from missirt import fuse_m, rho_from_labels, estimate_rho

r = RIDX; Y = YALL[r]; R = M[r].astype(bool)
C = {t: pickle.load(open(f's1_{t}.pkl', 'rb')) for t in ('rom', 'geo')}
cen = C['rom']['cen']
SPL = [f'cv{k}' for k in range(5)] + ['c1to2', 'c2to1']

def deep_cls(P1, P2): return (P1 > .5).astype(int) + (P2 > .5)

def wkappa(a, b):
    a = a.ravel(); b = b.ravel(); K = 3; O = np.zeros((K, K))
    for i, j in zip(a, b): O[i, j] += 1
    W = np.array([[(i - j) ** 2 for j in range(K)] for i in range(K)]) / (K - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * E).sum()

def icc_a1(x, y):
    X = np.c_[x, y].astype(float); n, k = X.shape; gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)

def metrics(pred, Yt):
    e = pred.sum(1) - Yt.sum(1)
    return dict(acc=float((pred == Yt).mean()), kappa=float(wkappa(Yt, pred)), mae=float(np.abs(e).mean()), bias=float(e.mean()),
                icc=float(icc_a1(Yt.sum(1), pred.sum(1))), within2=float((np.abs(e) <= 2).mean()))

def boot_diff(pa, pb, Yt, groups, B=2000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {u: np.flatnonzero(groups == u) for u in ug}
    ea = np.abs(pa.sum(1) - Yt.sum(1)); eb = np.abs(pb.sum(1) - Yt.sum(1)); aa = (pa == Yt).mean(1); ab = (pb == Yt).mean(1); dm = []; da = []
    for _ in range(B):
        s = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))]); dm.append(eb[s].mean() - ea[s].mean()); da.append(ab[s].mean() - aa[s].mean())
    return dict(mae=[float((eb - ea).mean())] + [float(x) for x in np.percentile(dm, [2.5, 97.5])],
                acc=[float((ab - aa).mean())] + [float(x) for x in np.percentile(da, [2.5, 97.5])])

# ---------------- per-split predictions on Xi'an -------------------------------------------------------------------
def split_preds(s):
    out = {}
    for t in ('rom', 'geo'):
        sp = C[t][s]; tr, te, P, oof = sp['tr'], sp['te'], sp['Pte'], sp['oof']
        sfx = '' if t == 'rom' else '+geo'
        out['RF' + sfx] = P.argmax(2)
        out['Stacking' + sfx] = stage2(oof, Y[tr], P)
        g1 = GPCM().fit(Y[tr]); a, b = fit_tau(g1, oof, Y[tr], mask=R[tr]); out['IRT-1D' + sfx] = fuse(g1, P, a, b, mask=R[te])[0].argmax(2)
        g2 = MGPCM(smooth=1.0).fit(Y[tr]); to, tc = fit_tau(g2, oof, Y[tr], mask=R[tr])
        out['IRT-2D' + sfx] = fuse(g2, P, to, tc, mask=R[te])[0].argmax(2)
        rho0 = rho_from_labels(Y[tr], R[tr]); rho = estimate_rho(g2, P, R[te], rho0, to, tc)
        out['MA-IRT' + sfx] = fuse_m(g2, P, R[te], rho, to, tc).argmax(2)
        if t == 'rom':
            # ablations of the latent fusion (tempering exponents fixed at the fitted values unless switched off)
            out['Abl: own evidence + population prior only'] = fuse(g2, P, to, 0.0, mask=R[te])[0].argmax(2)
            out['Abl: other items only (leave-one-out)'] = fuse(g2, P, 0.0, tc, mask=R[te])[0].argmax(2)
            out['Abl: untempered (tau=1)'] = fuse(g2, P, 1.0, 1.0, mask=R[te])[0].argmax(2)
            out['_tau'] = (to, tc)
    return out

if __name__ == '__main__':
    res = {'xian_cv': {}, 'transfer': {}, 'beijing': {}}
    per = {s: split_preds(s) for s in SPL}; print('fusion done', flush=True)
    # deep baselines
    z = np.load('work/rehab_cv_hier.npz'); htcv = deep_cls(z['P1'], z['P2'])
    zf = np.load('deep_cv_flat.npz'); flcv = deep_cls(zf['P1'], zf['P2'])
    lrcv = np.load('work/rehab_cv_lr.npy').astype(int)
    names = [k for k in per['cv0'] if not k.startswith('_')]
    cvp = {k: np.zeros_like(Y) for k in names}
    for k in range(5):
        te = C['rom'][f'cv{k}']['te']
        for n in names: cvp[n][te] = per[f'cv{k}'][n]
    cvp['LR'] = lrcv; cvp['Flat TCN'] = flcv; cvp['HT-Net'] = htcv
    for n, p in cvp.items():
        res['xian_cv'][n] = metrics(p, Y)
        if n != 'RF': res['xian_cv'][n]['vs_RF'] = boot_diff(cvp['RF'], p, Y, GRP)
    for n in ('MA-IRT', 'IRT-2D'):
        res['xian_cv'][n]['vs_Stacking'] = boot_diff(cvp['Stacking'], cvp[n], Y, GRP)
        res['xian_cv'][n]['vs_HT-Net'] = boot_diff(cvp['HT-Net'], cvp[n], Y, GRP)
    res['taus'] = {s: [float(per[s]['_tau'][0]), float(per[s]['_tau'][1])] for s in SPL}
    # severity strata (development definition: FMA27 < 40 vs >= 40)
    t = Y.sum(1); sev = t < 40
    res['xian_cv_strata'] = {n: dict(sev_mae=float(np.abs(p.sum(1) - t)[sev].mean()), mild_mae=float(np.abs(p.sum(1) - t)[~sev].mean()),
                                     sev_bias=float((p.sum(1) - t)[sev].mean()), mild_bias=float((p.sum(1) - t)[~sev].mean())) for n, p in cvp.items()}
    res['n'] = dict(xian_assess=int(len(Y)), xian_patients=int(len(np.unique(GRP))), sev=int(sev.sum()))
    # center transfer
    for s in ('c1to2', 'c2to1'):
        te = C['rom'][s]['te']; Yt = Y[te]; gt = GRP[te]; tt = Yt.sum(1); mild = tt >= 40
        P = {n: per[s][n] for n in names}
        for kind in ('flat', 'hier'):
            zz = np.load(f'deep_{s}_{kind}.npz'); P['Flat TCN' if kind == 'flat' else 'HT-Net'] = deep_cls(zz['P1'], zz['P2'])
        res['transfer'][s] = {}
        for n, p in P.items():
            m = metrics(p, Yt); m['mild_mae'] = float(np.abs(p.sum(1) - tt)[mild].mean()); m['sev_mae'] = float(np.abs(p.sum(1) - tt)[~mild].mean())
            if n != 'RF': m['vs_RF'] = boot_diff(P['RF'], p, Yt, gt)
            res['transfer'][s][n] = m
    # Beijing (locked)
    pla = PIDX; Yp = YALL[pla]; Rp = M[pla].astype(bool); pid = D.iloc[pla].pid.to_numpy(); tp = Yp.sum(1); sevp = tp < 40
    BJ = {}
    for t in ('rom', 'geo'):
        e = np.load(f'ext_probs_{t}.npz'); P, oof = e['P'], e['oof']; sfx = '' if t == 'rom' else '+geo'
        BJ['RF' + sfx] = P.argmax(2)
        BJ['RF' + sfx + ' + zero rule'] = np.where(Rp, P.argmax(2), 0)
        BJ['Stacking' + sfx] = stage2(oof, Y, P)
        P0 = P.copy(); P0[~Rp] = [0.98, 0.01, 0.01]
        BJ['Stacking' + sfx + ' + zero rule'] = np.where(Rp, stage2(oof, Y, P0), 0)
        g2 = MGPCM(smooth=1.0).fit(Y); to, tc = fit_tau(g2, oof, Y, mask=R)
        BJ['IRT-2D' + sfx] = fuse(g2, P, to, tc, mask=Rp)[0].argmax(2)
        rho0 = rho_from_labels(Y, R); rho = estimate_rho(g2, P, Rp, rho0, to, tc)
        BJ['MA-IRT' + sfx] = fuse_m(g2, P, Rp, rho, to, tc).argmax(2); res['beijing_rho_' + t] = [float(x) for x in rho]
    zf = np.load('work/dl_R2P_flat.npz'); zh = np.load('work/dl_R2P_hier.npz')
    BJ['Flat TCN'] = deep_cls(zf['P1'], zf['P2']); BJ['HT-Net'] = deep_cls(zh['P1'], zh['P2'])
    BJ['HT-Net + zero rule'] = np.where(Rp, BJ['HT-Net'], 0)
    res['beijing_true_rho'] = [float(((~Rp) & (Yp == k)).sum() / (Yp == k).sum()) for k in range(3)]
    res['beijing_missing'] = dict(sev=float((~Rp[sevp]).mean()), mild=float((~Rp[~sevp]).mean()), n_sev=int(sevp.sum()), n=int(len(tp)),
                                  missing_scored0=int(((~Rp) & (Yp == 0)).sum()), missing_total=int((~Rp).sum()),
                                  xian_missing_scored0=int(((~R) & (Y == 0)).sum()), xian_missing_total=int((~R).sum()))
    for n, p in BJ.items():
        m = metrics(p, Yp); e = p.sum(1) - tp
        m.update(sev_mae=float(np.abs(e)[sevp].mean()), sev_bias=float(e[sevp].mean()), mild_mae=float(np.abs(e)[~sevp].mean()), mild_bias=float(e[~sevp].mean()))
        if n != 'RF': m['vs_RF'] = boot_diff(BJ['RF'], p, Yp, pid)
        res['beijing'][n] = m
    np.savez('paper_preds.npz', **{('cv_' + k): v for k, v in cvp.items()}, **{('bj_' + k): v for k, v in BJ.items()})
    json.dump(res, open('results_paper.json', 'w'), indent=1)
    print('saved')
