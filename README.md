# Missingness-aware latent-trait fusion for automated FMA-UE assessment

This repository accompanies the manuscript **Missingness-aware latent-trait fusion of incomplete wearable sensor evidence for automated Fugl-Meyer upper-extremity assessment**, prepared for submission to *Biomedical Signal Processing and Control* (BSPC).

The proposed missingness-aware latent-trait fusion (MALT) method combines task-level evidence from wearable arm inertial measurement units and a sensor glove. A two-dimensional generalized partial credit model (GPCM) links proximal and distal motor ability, and leave-one-out inference combines each item's own evidence with contextual evidence from the other items. The recording indicator is modelled as a score-dependent observation, with missingness rates re-estimated in an unlabeled target cohort using an EM-style iteration. The repository also contains the comparative, robustness, uncertainty, and shortened-assessment experiments reported in the manuscript.

## Repository layout

| Directory | Contents |
|---|---|
| `prep/` | Raw signal reading (`xc_io.py`), REHAB assessment-to-patient linkage (`link_rehab.py`), item-level kinematic features (`feats.py`), resampled temporal sequences (`seqprep.py`), and correlation subspaces (`subspaces.py`). |
| `fusion/` | Data access (`common.py`), one-dimensional GPCM and leave-one-out inference (`irt.py`), two-dimensional GPCM (`mirt.py`), missingness-aware fusion (`missirt.py`), and the Transformer and graph convolutional fusion baselines (`learned_fusion.py`). |
| `experiments/` | Experiment scripts, temporal baselines (`htnet.py`), and time-stamped SHA-256 configuration lock records (`LOCK_*.txt`). |
| `figures/` | Scripts for generating the data-driven figures (Figures 2–5). Figure 1 is a conceptual schematic supplied with the manuscript rather than an analytical output. |

## Data access and preparation

Place the required files in the following locations, relative to the repository root:

| Path | Contents |
|---|---|
| `data/rehab/Rehab_assessment/` | Raw assessment folder of the REHAB dataset, containing `d01_raw_data/` and `label.npy`. |
| `data/rehab/brs.xlsx` | REHAB clinical score table (sheet `data_set1`), including assessment dates and FMA-UE item scores. |
| `data/rehab/Rehabilitation_data.xlsx` | Development trial case report forms used to link assessments to patient identifiers and centers for patient-grouped evaluation. Obtain access through the appropriate data custodian if this file is not included in your authorized dataset. |
| `data/beijing/01*/` | Beijing test cohort files. These data are not publicly released and are available from the corresponding author on reasonable request, subject to applicable data-access and ethics requirements. |

The development dataset is described in Lv et al., *Scientific Data* (2026), [doi:10.1038/s41597-026-07802-2](https://doi.org/10.1038/s41597-026-07802-2). The released assessments are linked to the trial case report forms to obtain patient groups and center assignments. The analysis includes 222 development assessments from 119 patients and 68 Beijing assessments from 37 patients.

The code retains the device/data item numbering. Figure 2 converts the relevant device item numbers to the standard FMA-UE numbering used in the manuscript (`DEV2STD` in `figures/figs_data.py`). The instrumented outcome comprises 27 items, with a total score of 0–54; the six remaining items are scored by the therapist.

**Access limitation:** the full pipeline currently expects both the development and Beijing datasets. Beijing data are required for the external-cohort analyses and for running the pipeline as supplied; the private cohort cannot be reproduced without authorized access. Do not commit identifiable patient records or restricted source files to the public repository.

## Installation and execution

Create a Python environment and install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Run the complete pipeline from the repository root using a POSIX-compatible shell (for example, Git Bash on Windows):

```sh
sh run_all.sh
```

The script sets `PYTHONPATH=prep:fusion:experiments` and executes the preparation and experiment stages in order. Intermediate caches and outputs are written to the repository root and `work/`. The full pipeline was run on two CPU cores; a GPU is not required.

## Reproducing the manuscript results

The following table maps the experiment scripts and generated outputs to the current BSPC manuscript and supplementary material. A dagger (†) indicates analyses that require the Beijing cohort; some scripts combine development and external-cohort analyses.

| Analysis | Script(s) | Generated output(s) | BSPC manuscript location |
|---|---|---|---|
| Patient linkage and signal preparation | `prep/link_rehab.py`, `prep/feats.py`, `prep/seqprep.py`, `prep/subspaces.py` | `work/rehab_link.csv`, `work/feats.pkl`, `work/seqs.npz`, `work/subspaces.pkl` | Sections 3.2 and 4.1–4.2; Supplementary Note S4 |
| Item-level random forest probabilities and inner out-of-fold evidence | `experiments/cache_s1.py` (add `geo` for coordination-geometry features) | `s1_rom.pkl`, `s1_geo.pkl` | Sections 3.2–3.4 and 4.2 |
| Logistic regression baseline | `experiments/lr_cv.py` | `work/rehab_cv_lr.npy` | Table 2 |
| Temporal baselines: flat TCN and HT-Net | `experiments/rehab_cv.py`, `experiments/deep_extra.py`, `experiments/dl_run.py`† | `work/rehab_cv_hier.npz`, `work/deep_*.npz`, `work/dl_*.npz` | Tables 2, 4 and 5 |
| Locked Beijing fusion evaluation † | `experiments/final_ext.py`, `experiments/final_ext2.py` | `ext_probs_*.npz`, `res_final_ext*.json` | Table 5; Supplementary Table S6 |
| Main fusion results and evaluation † | `experiments/eval_all.py` | `results_paper.json`, `paper_preds.npz` | Tables 2–5; Figure 4 |
| Transformer and graph convolutional network | `experiments/exp_learned.py`† | `results_learned.json` | Table 6 |
| Learned fusers with MCAR, MNAR and mixed augmentation | `experiments/exp_aug.py transformer`, `experiments/exp_aug.py gcn`† | `res_aug_transformer.json`, `res_aug_gcn.json` | Table 6; Supplementary Table S5 |
| Deployment conditions: small unlabeled batches; TENT and entropy-based model selection | `experiments/exp_batch.py malt`, `experiments/exp_batch.py gcn`, `experiments/exp_batch.py transformer`† | `res_batch_malt.json`, `res_batch_gcn.json`, `res_batch_transformer.json` | Section 5.6; Supplementary Table S7 |
| Classical fusion, calibration, sensor failure and extra ablations | `experiments/exp_unc.py`, `experiments/exp_unc_learned.py`, `experiments/sum_unc.py` | `res_unc.json` | Tables 3 and 7; Supplementary Table S1; Figure 5B |
| Shortened and adaptive assessment | `experiments/exp_short.py`, `experiments/sum_short.py` | `res_short.json` | Section 5.8; Supplementary Table S8; Figure 5A |
| Controlled missingness | `experiments/exp_sim.py` | `res_sim.json` | Section 5.4; Figure 3 |
| Repeated patient-grouped cross-validation (partitions 2–5) | `experiments/exp_repcv.py 1 2 3 4`, `experiments/sum_repcv.py` | `repcv_*.pkl`, `res_repcv.json` | Section 5.1; Supplementary Table S2 |
| Few-shot site calibration between the Xi’an hospitals | `experiments/exp_fewshot_xian.py c1to2 1,2,3,5,8,12 10` (and `c2to1`) | `fs_<split>_<m>.json` | Section 5.3; Supplementary Table S3 |
| Numerical illustration of Proposition 2 | `experiments/prop2_check.py` | `res_prop2.json` | Section 3.6; Supplementary Notes S1 and S5 |
| Item hierarchy | `experiments/hier_cv.py` | `hier_cv.json` | Figure 2 |
| Data-driven figures | `figures/figs_data.py`, `figures/fig4.py`, `figures/fig5.py` | `fig*.png`, `fig*.pdf` | Figures 2–5 |

## Neural baselines and reproducibility

The flat TCN, HT-Net, Transformer and GCN are implemented in PyTorch. Initial weights, data augmentation, and batch order use seeded NumPy generators. Small floating-point differences across platforms, package versions, or thread counts may affect individual predictions. The manuscript experiments were run with PyTorch 2.15 and one OpenMP thread per process (`OMP_NUM_THREADS=1`).

The reported environment was Python 3.13, NumPy 2.5, SciPy 1.18, scikit-learn 1.9, pandas 3.0, Matplotlib 3.11, openpyxl 3.1, and PyTorch 2.15. See `requirements.txt` for dependency specifications.

## Beijing configuration locks

The files `experiments/LOCK_A.txt`, `experiments/LOCK_FINAL.txt`, and `experiments/LOCK_MA.txt` contain time-stamped SHA-256 records for the configurations frozen at the corresponding stages of the Beijing evaluation. The locked fusion files `exp_a.py`, `irt.py`, `mirt.py`, `missirt.py`, `final_ext.py`, and `final_ext2.py` match their recorded hashes. `fusion/common.py` was modified after the lock to adjust file paths for release; its current hash therefore differs from the lock record. The neural baselines are not covered by these locks. See Section 4.2 and Supplementary Table S6 for the two-stage evaluation protocol and the distinction between locked and post hoc comparisons.

## License

A code reuse license has not yet been specified. Please contact the authors regarding reuse permissions. Data access and reuse are governed separately by the applicable dataset terms and ethical approvals.
