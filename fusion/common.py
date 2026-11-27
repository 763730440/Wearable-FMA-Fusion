import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'work')
import numpy as np, pandas as pd
from xc_io import ITEMS, HAS_BEIJING

D = pd.read_pickle('work/feats.pkl')
S = np.load('work/seqs.npz'); ARM, HAND, M = S['ARM'], S['HAND'], S['M']
FC = {it: [c for c in D.columns if c.startswith(f'i{it}_')] for it in ITEMS}
YALL = D[[f'y{i}' for i in ITEMS]].to_numpy().astype(int)
RIDX = np.flatnonzero((D.cohort == 'REHAB').to_numpy())
PIDX = np.flatnonzero((D.cohort == 'PLA').to_numpy())
L = pd.read_csv('work/rehab_link.csv').sort_values('row').reset_index(drop=True)
GRP = L.subj.to_numpy()          # patient id for each REHAB row (same order as RIDX)
VISIT = L.visit.to_numpy()

# --- geometry -----------------------------------------------------------------------------
def corr_subspace(X, mask_ok, p):
    """X: (T, d) one recording. Returns d x p orthonormal basis of leading correlation subspace and normalized eigenvalues."""
    d = X.shape[1]
    if not mask_ok: return None, None
    Xc = X - X.mean(0)
    sd = Xc.std(0); sd[sd < 1e-8] = 1.0
    Z = Xc / sd
    C = Z.T @ Z / max(len(Z) - 1, 1) + 1e-3 * np.eye(d)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    return V[:, :p], w / w.sum()

def all_subspaces(idx, block, p):
    """For rows idx: dict item_index -> (list of bases or None), eig spectra (n, k)."""
    A = ARM if block == 'arm' else HAND
    U = np.empty((len(idx), len(ITEMS)), object); E = np.full((len(idx), len(ITEMS), 3), np.nan)
    for a, i in enumerate(idx):
        for j in range(len(ITEMS)):
            u, w = corr_subspace(A[i, j], bool(M[i, j]) and np.isfinite(A[i, j]).all() and A[i, j].std() > 0, p)
            U[a, j] = u
            if w is not None: E[a, j] = w[:3]
    return U, E

def proj_mean(bases, p):
    """Extrinsic (projection) mean subspace of a list of bases."""
    bases = [b for b in bases if b is not None]
    if not bases: return None
    Pm = sum(b @ b.T for b in bases) / len(bases)
    w, V = np.linalg.eigh(Pm)
    return V[:, ::-1][:, :p]

def pdist(u, v):
    if u is None or v is None: return np.nan
    return np.linalg.norm(u @ u.T - v @ v.T) / np.sqrt(2)
