#!/usr/bin/env bash
# =====================================================================
# run_all.sh -- full xApp-Guard evaluation pipeline
# ---------------------------------------------------------------------
# Runs the four experiments and regenerates the figures. Every number
# in the article's Table II and Figures 3 and 4 is produced here; none
# is hand-written.
#
#   1. detection-accuracy experiment  (A1, A2, A3 -- 5 seeds each)
#   2. scaling experiment             (CPU & O1 vs #xApps)
#   3. ROC threshold-sweep experiment (TPR/FPR vs WATCH percentile)
#   4. feature-importance experiment  (benign-ablation attribution)
#   5. figure generation              (from the measured CSVs)
#
# Total wall-clock: ~10-12 min on a modern 8-core laptop.
# =====================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "==> [1/5] detection-accuracy experiment (A1, A2, A3)"
python3 scripts/run_detection_experiment.py A1
python3 scripts/run_detection_experiment.py A2
python3 scripts/run_detection_experiment.py A3

echo
echo "==> [2/5] scaling experiment (CPU & O1 bandwidth)"
python3 scripts/run_scaling_experiment.py

echo
echo "==> [3/5] ROC threshold-sweep experiment"
python3 scripts/run_roc_experiment.py A1 A2
python3 scripts/run_roc_experiment.py A3

echo
echo "==> [4/5] feature-importance experiment"
python3 scripts/run_importance_experiment.py

echo
echo "==> [5/5] generating figures from measured results"
python3 scripts/make_figures.py

echo
echo "==> done. Artefacts in results/:"
ls -1 results/
