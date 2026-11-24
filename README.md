# Missingness-aware latent-trait fusion for automated FMA-UE scoring

Code for the manuscript *Missingness-aware latent-trait fusion of task-level wearable evidence for automated Fugl-Meyer upper-extremity scoring* (under preparation for Information Fusion).

Item classifiers give evidence about each FMA-UE task. A two-dimensional generalized partial credit model (proximal and distal ability, nonparametric joint prior) fuses that evidence across tasks with leave-one-out inference. The presence of a recording is modelled as a second, score-dependent observation, re-estimated in a new site by EM without labels.

## Layout

| Folder | Contents |
|---|---|
| `prep/` | Reading the raw recordings (`xc_io.py`), kinematic features per item (`feats.py` → `work/feats.pkl`), resampled sequences for the deep baselines (`seqprep.py` → `work/seqs.npz`), a small numpy autograd (`nn.py`) |
| `fusion/` | Data access (`common.py`), 1-D GPCM and leave-one-out fusion with tempering (`irt.py`), 2-D between-item GPCM (`mirt.py`), missingness-aware fusion and label-free EM for the missingness rates (`missirt.py`), Transformer and GCN fusion baselines (`learned_fusion.py`, `nnx.py`) |
| `experiments/` | One script per analysis (see below); `LOCK_*.txt` hold the time-stamped hashes that froze the configurations before the Beijing test |
| `figures/` | Figure scripts |

## Data

- **Development data (Xi'an, 222 assessments, 119 patients):** the public REHAB dataset (Lv et al., *Sci Data* 2026). Place the raw assessment folder in `data/rehab/Rehab_assessment/` and the clinical tables in `data/rehab/`.
- **Beijing test data:** not public. The Beijing scripts (`final_ext*.py`, `dl_run.py`) cannot be run without it.

All scripts run from the repository root with `prep/`, `fusion/` and `experiments/` on `PYTHONPATH`. Intermediate files are written to `work/`.

## Reproducing the results

| Step | Script | Output | Manuscript |
|---|---|---|---|
| Features and sequences | `prep/feats.py`, `prep/seqprep.py` | `work/feats.pkl`, `work/seqs.npz` | Section 3.2 |
| Item classifiers (all splits, inner out-of-fold) | `experiments/cache_s1.py` (`geo` for geometry features) | `s1_rom.pkl`, `s1_geo.pkl` | Section 3.2 |
| LR baseline | `experiments/lr_cv.py` | `rehab_cv_lr.npy` | Table 2 |
| Deep baselines | `experiments/rehab_cv.py`, `deep_extra.py`, `dl_run.py` | `*_flat.npz`, `*_hier.npz` | Tables 2, 4, 5 |
| Main results | `experiments/eval_all.py` | `results_paper.json` | Tables 2-5, Fig 4 |
| Controlled missingness | `experiments/exp_sim.py` | `res_sim.json` | Section 5.4, Fig 3 |
| Transformer and GCN | `experiments/exp_learned.py` | `results_learned.json` | Table 6 |
| Classical fusion, uncertainty, sensor-block failure, extra ablations | `experiments/exp_unc.py`, `exp_unc_learned.py`, `sum_unc.py` | `res_unc.json` | Tables 3, 7, 8; Sections 5.7-5.9 |
| Shortened (adaptive) assessment | `experiments/exp_short.py`, `sum_short.py` | `res_short.json` | Table 9, Fig 5 |
| Few-shot site calibration between the Xi'an hospitals | `experiments/exp_fewshot_xian.py` | `res_fewshot_xian.json` | Section 5.3 |
| Repeated cross-validation | `experiments/exp_repcv.py` | `repcv_*.pkl` | Robustness analysis |
| Proposition 2 illustration | `experiments/prop2_check.py` | `res_prop2.json` | Section 5.7 |
| Figures | `figures/*.py` | `fig*.png/pdf` | Figs 1-5 |

## Environment

Python 3.13; numpy 2.5, scipy 1.18, scikit-learn 1.9, pandas 3.0, matplotlib 3.11. The Transformer, GCN and temporal networks use the numpy autograd in `prep/nn.py`, so no deep-learning framework is needed. Everything runs on a CPU; the full pipeline takes a few hours on two cores.

## License

To be decided by the authors before release.
