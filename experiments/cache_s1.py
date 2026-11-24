"""Cache stage-1 RF item probabilities for the Xi'an development experiments.
Splits: pooled patient-grouped 5-fold CV ('cv'), and center transfer ('c1to2', 'c2to1').
For every split: test probabilities from a model fitted on the training part, and inner grouped out-of-fold
probabilities on the training part (for fitting fusion/calibration parameters without touching test labels)."""
import pickle, sys
from exp_a import stage1, r, grp, Y
from common import *
from sklearn.model_selection import GroupKFold

use_geo = len(sys.argv) > 1 and sys.argv[1] == 'geo'
crf = pd.read_excel('data/rehab/Rehabilitation_data.xlsx').set_index('SUBJID')
cen = crf.loc[GRP, 'CENTER'].to_numpy()
splits = [('cv%d' % k, tr, te) for k, (tr, te) in enumerate(GroupKFold(5).split(r, groups=grp))]
splits += [('c1to2', np.flatnonzero(cen == 1), np.flatnonzero(cen == 2)), ('c2to1', np.flatnonzero(cen == 2), np.flatnonzero(cen == 1))]
out = {'cen': cen}
for name, tr, te in splits:
    Pte = stage1(r[tr], r[te], use_geo)
    oof = np.zeros((len(tr), len(ITEMS), 3))
    for itr, ite in GroupKFold(4).split(tr, groups=grp[tr]):
        oof[ite] = stage1(r[tr][itr], r[tr][ite], use_geo)
    out[name] = dict(tr=tr, te=te, Pte=Pte, oof=oof)
    print(name, flush=True)
pickle.dump(out, open('s1_%s.pkl' % ('geo' if use_geo else 'rom'), 'wb'))
