"""Leading correlation subspaces of every recording (arm: 3-D, hand: 2-D) -> work/subspaces.pkl (used by the geometry features)."""
import pickle, numpy as np
from common import D, all_subspaces
idx = np.arange(len(D))
UA, EA = all_subspaces(idx, 'arm', 3); UH, EH = all_subspaces(idx, 'hand', 2)
G = dict(UA=UA, EA=EA, UH=UH, EH=EH)
pickle.dump(G, open('work/subspaces.pkl', 'wb'))
pickle.dump(G, open('subspaces.pkl', 'wb'))   # exp_a.py (hash-locked) reads ./subspaces.pkl
print('subspaces', UA.shape, UH.shape)
