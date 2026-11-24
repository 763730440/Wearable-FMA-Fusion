"""LR baseline: per-item multinomial logistic regression on the kinematic features, patient-grouped 5-fold CV."""
import numpy as np
from common import *
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
r = RIDX; Y = YALL[r]; pred = np.zeros_like(Y)
for tr, te in GroupKFold(5).split(r, groups=GRP):
    for j, it in enumerate(ITEMS):
        X = D.iloc[r][FC[it]].to_numpy(float); y = Y[tr, j]
        if len(np.unique(y)) == 1: pred[te, j] = y[0]; continue
        m = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')).fit(X[tr], y)
        pred[te, j] = m.predict(X[te])
np.save('work/rehab_cv_lr.npy', pred)
print('acc', (pred == Y).mean(), 'mae', np.abs(pred.sum(1) - Y.sum(1)).mean())
