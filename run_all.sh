#!/bin/sh
# Runs the full pipeline from the repository root. Steps marked (Beijing) need data/beijing/.
set -e
export PYTHONPATH=prep:fusion:experiments OMP_NUM_THREADS=1
mkdir -p work
# The Beijing cohort is private. Without data/beijing/01*, the Beijing steps are skipped and every
# development (Xi'an) analysis still runs from the public REHAB data.
if ls -d data/beijing/01* >/dev/null 2>&1; then BEIJING=1; else BEIJING=0; echo "data/beijing not found: skipping Beijing steps"; fi
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
if [ $BEIJING = 1 ]; then python experiments/dl_run.py; fi            # (Beijing)
# Beijing locked test of the fusion methods
if [ $BEIJING = 1 ]; then
  python experiments/final_ext.py              # (Beijing)
  python experiments/final_ext2.py             # (Beijing)
fi
# main results, learned fusion, uncertainty
python experiments/eval_all.py
python experiments/exp_learned.py
for k in gcn transformer; do python experiments/exp_aug.py $k; done
for k in malt gcn transformer; do python experiments/exp_batch.py $k; done
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
# Data-driven figures only. Figure 1 is supplied separately as a conceptual schematic.
python figures/figs_data.py
python figures/fig5.py
if [ $BEIJING = 1 ]; then python figures/fig4.py; fi   # Figure 4 shows the Beijing test
