"""How much of each item's clinician score is predictable from the clinician scores of the other items through the latent
trait model (patient-grouped 5-fold CV, true labels as evidence for the other items)."""
import json, pickle
from common import *
from irt import GPCM, fuse
from mirt import MGPCM, TH
from sklearn.model_selection import GroupKFold
r = RIDX; Y = YALL[r]; out = {'1D': np.zeros_like(Y), '2D': np.zeros_like(Y), 'prior_only': np.zeros_like(Y)}
for tr, te in GroupKFold(5).split(r, groups=GRP):
    E = np.eye(3)[Y[te]] * 0.98 + 0.0067
    for name, g in (('1D', GPCM().fit(Y[tr])), ('2D', MGPCM(smooth=1.0).fit(Y[tr]))):
        out[name][te] = fuse(g, E, 0.0, 1.0)[0].argmax(2)
    out['prior_only'][te] = np.bincount(Y[tr].ravel(), minlength=3).argmax() if False else np.array([np.bincount(Y[tr][:, j], minlength=3).argmax() for j in range(Y.shape[1])])[None]
res = {k: dict(acc=float((v == Y).mean()), per_item=[float(x) for x in (v == Y).mean(0)]) for k, v in out.items()}
g = MGPCM(smooth=1.0).fit(Y); res['prior2d'] = g.w.reshape(31, 31).tolist(); res['items'] = ITEMS
res['disc'] = np.exp(g.a).tolist()
json.dump(res, open('hier_cv.json', 'w')); print({k: round(v['acc'], 3) for k, v in res.items() if isinstance(v, dict)})
