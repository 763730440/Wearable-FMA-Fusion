# Missingness-aware latent-trait fusion for automated FMA-UE scoring

Code for the manuscript *Missingness-aware latent-trait fusion of task-level wearable evidence for automated Fugl-Meyer upper-extremity scoring* (submitted to Information Fusion).

Item classifiers give evidence about each FMA-UE task. A two-dimensional generalized partial credit model (proximal and distal ability, nonparametric joint prior) fuses that evidence across tasks with leave-one-out inference. The presence of a recording is modelled as a second, score-dependent observation, re-estimated in a new site by EM without labels.

## Layout

| Folder | Contents |
|---|---|
| `prep/` | Reading the raw recordings (`xc_io.py`); linking the REHAB assessments to trial patients (`link_rehab.py` → `work/rehab_link.csv`); kinematic features per item (`feats.py` → `work/feats.pkl`); resampled sequences for the temporal baselines (`seqprep.py` → `work/seqs.npz`); correlation subspaces for the geometry features (`subspaces.py` → `work/subspaces.pkl`) |
| `fusion/` | Data access (`common.py`), 1-D GPCM and leave-one-out fusion with tempering (`irt.py`), 2-D between-item GPCM (`mirt.py`), missingness-aware fusion and label-free EM for the missingness rates (`missirt.py`), Transformer and GCN fusion baselines in PyTorch (`learned_fusion.py`) |
| `experiments/` | One script per analysis (see below); flat TCN and HT-Net in PyTorch (`htnet.py`); `LOCK_*.txt` hold the time-stamped hashes that froze the configurations before the Beijing test |
| `figures/` | Figure scripts |

## Data

Place the data as follows (paths are relative to the repository root):

| Path | Contents |
|---|---|
| `data/rehab/Rehab_assessment/` | Raw assessment folder of the public REHAB dataset (Lv et al., *Sci Data* 2026): `d01_raw_data/`, `label.npy` |
| `data/rehab/brs.xlsx` | REHAB clinical score table (sheet `data_set1`: 33 FMA-UE item scores, Brunnstrom stages, assessment dates) |
| `data/rehab/Rehabilitation_data.xlsx` | Case report forms of the development trial (patient IDs, center, visit dates, item scores) |
| `data/beijing/01*/` | Beijing test data (not public; available from the corresponding author on reasonable request). The Beijing scripts (`final_ext*.py`, `dl_run.py`) and the Beijing parts of `eval_all.py`, `exp_learned.py`, `exp_aug.py` and `exp_batch.py` cannot be run without it. |

## Reproducing the results

`run_all.sh` runs every step in order from the repository root (`PYTHONPATH=prep:fusion:experiments`). Steps marked † need the Beijing data. Outputs are written to `work/` or to the repository root as listed.

| Step | Script | Output | Manuscript |
|---|---|---|---|
| Patient linkage, features, sequences, subspaces | `prep/link_rehab.py`, `prep/feats.py`, `prep/seqprep.py`, `prep/subspaces.py` | `work/rehab_link.csv`, `work/feats.pkl`, `work/seqs.npz`, `work/subspaces.pkl` | Section 3.2 |
| Item classifiers (all splits, inner out-of-fold) | `experiments/cache_s1.py` (`geo` for geometry features) | `s1_rom.pkl`, `s1_geo.pkl` | Section 3.2 |
| LR baseline | `experiments/lr_cv.py` | `work/rehab_cv_lr.npy` | Table 2 |
| Temporal baselines (flat TCN, HT-Net) | `experiments/rehab_cv.py`, `deep_extra.py`, `dl_run.py`† | `work/rehab_cv_hier.npz`, `work/deep_*.npz`, `work/dl_*.npz` | Tables 2, 4, 5 |
| Beijing locked test of the fusion methods † | `experiments/final_ext.py`, `final_ext2.py` | `ext_probs_*.npz`, `res_final_ext*.json` | Table 5 |
| Main results | `experiments/eval_all.py` | `results_paper.json`, `paper_preds.npz` | Tables 2-5, Fig 4 |
| Transformer and GCN | `experiments/exp_learned.py` | `results_learned.json` | Table 6 |
| Transformer and GCN with missing-data augmentation (MCAR, MNAR, mixed) | `experiments/exp_aug.py transformer`, `exp_aug.py gcn` | `res_aug_transformer.json`, `res_aug_gcn.json` | Table 6 (augmented rows); Supplementary Table S5 |
| Deployment conditions: latent fusion with small unlabeled batches; learned fusers with TENT and entropy selection | `experiments/exp_batch.py malt`, `exp_batch.py gcn`, `exp_batch.py transformer` | `res_batch_malt.json`, `res_batch_gcn.json`, `res_batch_transformer.json` | Supplementary Table S7 |
| Classical fusion, uncertainty, sensor-block failure, extra ablations | `experiments/exp_unc.py`, `exp_unc_learned.py`, `sum_unc.py` | `res_unc.json` | Tables 3, 7; Supplementary Table S1; Fig 5B |
| Shortened (adaptive) assessment | `experiments/exp_short.py`, `sum_short.py` | `res_short.json` | Table 8, Fig 5A |
| Controlled missingness | `experiments/exp_sim.py` | `res_sim.json` | Section 5.4, Fig 3 |
| Repeated cross-validation (partitions 2-5) | `experiments/exp_repcv.py 1 2 3 4`, `sum_repcv.py` | `repcv_*.pkl`, `res_repcv.json` | Supplementary Table S2, Section 5.11 |
| Few-shot site calibration between the Xi'an hospitals | `experiments/exp_fewshot_xian.py c1to2 1,2,3,5,8,12 10` (and `c2to1`) | `fs_<split>_<m>.json` | Supplementary Table S3, Section 5.3 |
| Proposition 2 illustration | `experiments/prop2_check.py` | `res_prop2.json` | Section 5.7 |
| Item hierarchy | `experiments/hier_cv.py` | `hier_cv.json` | Fig 2 |
| Figures | `figures/*.py` | `fig*.png/pdf` | Figs 2-5 |

Figure 2A reads `hier_cv.json`, produced by `experiments/hier_cv.py` (item hierarchy: each item predicted from the clinician scores of the other items).

## Neural baselines

The flat TCN, HT-Net, Transformer and GCN baselines are implemented in PyTorch (`experiments/htnet.py`, `fusion/learned_fusion.py`) and run on a CPU. Initial weights are drawn from a seeded NumPy generator and augmentation and batch order use the same generator, so a given seed starts from the same weights on every machine; small floating-point differences between machines, PyTorch versions or thread counts can still change individual predictions slightly. The manuscript results were produced with this code, PyTorch 2.15, one thread per process (`OMP_NUM_THREADS=1`).

## Configuration locks

`experiments/LOCK_A.txt`, `LOCK_MA.txt` and `LOCK_FINAL.txt` record SHA-256 hashes of the fusion code taken before the Beijing data were analysed. `exp_a.py`, `irt.py`, `mirt.py`, `missirt.py`, `final_ext.py` and `final_ext2.py` match their locked hashes. `fusion/common.py` was edited afterwards when the code was prepared for release (file paths), so its hash differs from the one recorded in `LOCK_FINAL.txt`. The neural baselines are not covered by the locks.

## Environment

Python 3.13; numpy 2.5, scipy 1.18, scikit-learn 1.9, pandas 3.0, matplotlib 3.11, openpyxl 3.1, PyTorch 2.15 (CPU is sufficient). See `requirements.txt`. The full pipeline takes about six hours on two CPU cores.

## License

To be decided by the authors before release.
