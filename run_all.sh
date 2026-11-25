#!/bin/sh
# Runs the full pipeline from the repository root. Steps marked (Beijing) need data/beijing/.
set -e
export PYTHONPATH=prep:fusion:experiments OMP_NUM_THREADS=1
mkdir -p work
# data preparation
python prep/link_rehab.py
python prep/feats.py
python prep/seqprep.py
python prep/subspaces.py
# item classifiers
python experiments/cache_s1.py
python experiments/cache_s1.py geo
# baselines
python experiments/lr_cv.py
python experiments/rehab_cv.py
python experiments/deep_extra.py
python experiments/dl_run.py                 # (Beijing)
# Beijing locked test of the fusion methods
python experiments/final_ext.py              # (Beijing)
python experiments/final_ext2.py             # (Beijing)
# main results, learned fusion, uncertainty
python experiments/eval_all.py
python experiments/exp_learned.py
for k in gcn transformer; do python experiments/exp_aug.py $k; done   # learned fusers with missing-data augmentation
for k in malt gcn transformer; do python experiments/exp_batch.py $k; done   # deployment conditions: batch size, TENT, entropy selection
python experiments/exp_unc.py cv0 cv1 cv2 cv3 cv4 c1to2 c2to1
python experiments/exp_unc_learned.py
python experiments/sum_unc.py
# shortened assessment, controlled missingness, repeated CV, few-shot calibration, Proposition 2
python experiments/exp_short.py cv0 cv1 cv2 cv3 cv4 c1to2 c2to1
python experiments/sum_short.py
python experiments/exp_sim.py
python experiments/exp_repcv.py 1 2 3 4
python experiments/sum_repcv.py
for d in c1to2 c2to1; do python experiments/exp_fewshot_xian.py $d 1,2,3,5,8,12 10; done
python experiments/prop2_check.py
python experiments/hier_cv.py
# figures
for f in figures/fig1.py figures/figs_data.py figures/fig4.py figures/fig5.py; do python $f; done
