"""Link the 222 REHAB assessments to trial patients via the case report forms -> work/rehab_link.csv
(exact agreement on all 33 FMA-UE item scores, ties broken by date proximity to the baseline TINDAT / discharge VISDAT1).
Same procedure as in the companion validation study."""
import numpy as np, pandas as pd
crf = pd.read_excel('data/rehab/Rehabilitation_data.xlsx')
xl = pd.read_excel('data/rehab/brs.xlsx', sheet_name='data_set1', header=None).iloc[5:227].reset_index(drop=True)
X = np.array([[xl.iloc[i, k + 1] for k in range(1, 34)] for i in range(222)], float)
dt = pd.to_datetime(xl.iloc[:, 55].astype(str), format='%Y-%m-%d %H-%M-%S', errors='coerce')
B = crf[[f'QSQ{i:02d}' for i in range(1, 34)]].to_numpy(float); H = crf[[f'QSH{i:02d}' for i in range(1, 34)]].to_numpy(float)
bd = pd.to_datetime(crf.TINDAT); hd = pd.to_datetime(crf.VISDAT1)
rows, used = [], set()
for i in range(222):
    cands = []
    for j in range(len(crf)):
        for tag, Mx, dd in (('pre', B, bd), ('post', H, hd)):
            if np.isnan(Mx[j]).all(): continue
            mism = int(np.nansum(Mx[j] != X[i]))
            gap = abs((dt[i].normalize() - dd[j]).days) if pd.notna(dd[j]) and pd.notna(dt[i]) else 999
            cands.append((mism, gap, (crf.SUBJID[j], tag)))
    cands.sort(key=lambda c: (c[0], c[1]))
    best = next((c for c in cands if c[0] == cands[0][0] and c[2] not in used), cands[0])   # prefer an unused patient-visit among ties
    used.add(best[2]); rows.append((i, best[2][0], best[2][1], best[0], best[1]))
L = pd.DataFrame(rows, columns=['row', 'subj', 'visit', 'mismatch', 'daygap'])
L.to_csv('work/rehab_link.csv', index=False)
print('assessments', len(L), 'patients', L.subj.nunique(), 'exact item match', (L.mismatch == 0).mean())
