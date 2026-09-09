#!/usr/bin/env bash
# run_all.sh - reproduce every Tier A number and figure from a clean clone.
#
#   bash run_all.sh
#
# Tier A needs only what this repository ships (data/) plus requirements.txt.
# It trains three small MLPs on CPU; expect tens of minutes. Nothing here reads
# raw repertoire files.
#
# Tier B (pipeline/stage01..stage03a, emerson_build, clinical_build, vj_usage,
# the diversity half of baselines.py) regenerates the matrices in data/ from the
# raw immuneACCESS TSVs and needs TTIME_RAW_* plus a cluster. It is NOT run here.
#
# Tier C (Figure 3 panels a-d) needs h2o; see README "Figure 3 provenance".
#
# Outputs go to results/revision/ by default. To compare a fresh run against the
# shipped artefacts without overwriting them:
#
#   TTIME_REVISION_OUT=/tmp/ttime_check bash run_all.sh
#
set -euo pipefail
cd "$(dirname "$0")"

OUT="${TTIME_REVISION_OUT:-$PWD/results/revision}"
export TTIME_REVISION_OUT="$OUT"
export TTIME_WORK="${TTIME_WORK:-$PWD/workspace}"
export PYTHONHASHSEED=0
# C13: three Tier A scripts call plt.show(). The default backend here is interactive
# (QtAgg), which blocks forever in an unattended run - force a headless backend.
export MPLBACKEND="${MPLBACKEND:-Agg}"
PY="${PYTHON:-python}"
mkdir -p "$OUT"

step () { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "A1  primary model            -> $OUT/test_preds.csv"
$PY analysis/revision/01_train_primary_model.py

# Scripts 02-06 write to the current directory (C9), so they run with cwd = $OUT.
step "A2  external zero-shot + recalibration"
( cd "$OUT" && $PY "$OLDPWD/analysis/revision/02_external_zeroshot_recalibration.py" )

step "A3  sex-gap analysis"
( cd "$OUT" && $PY "$OLDPWD/analysis/revision/03_sex_gap_analysis.py" )

step "A4  clinical AAR"
( cd "$OUT" && $PY "$OLDPWD/analysis/revision/04_clinical_aar_pipeline.py" )

step "A5  all-categories exploration"
( cd "$OUT" && $PY "$OLDPWD/analysis/revision/05_all_categories_exploration.py" )

step "A6  Figure 2 (panels a-i)"
( cd "$OUT" && $PY "$OLDPWD/analysis/revision/06_figure_full_a_to_i.py" )

step "A7  CMV serostatus"
( cd "$OUT" && $PY "$OLDPWD/analysis/revision/08_cmv_seropositivity.py" )

step "A8  MAIT / physicochemical classifier"
$PY pipeline/mait_classifier.py

step "A9  interpretable baselines (Ridge/ElasticNet; diversity needs Tier B)"
$PY pipeline/baselines.py || echo "[!] baselines.py incomplete - diversity baselines need raw TSVs"

step "A10 Figures 3e-f and 4b"
$PY pipeline/figures_3_4.py

# C5: 13/14 now resolve their inputs from anywhere, so they run in $OUT like 02-06
# and their outputs land there instead of inside data/.
step "A11 epitope database harmonisation"
( cd "$OUT" && $PY "$OLDPWD/analysis/annotation/13_epitope_database_harmonization.py" )

step "A12 TCR-epitope age alignment"
( cd "$OUT" && $PY "$OLDPWD/analysis/annotation/14_tcr_epitope_age_alignment.py" )

step "A13 Figure 4 landscape"
( cd "$OUT" && $PY "$OLDPWD/analysis/annotation/Figure4_Landscape_Analysis.py" )

step "A14 assemble results/numbers.json"
$PY analysis/revision/07_build_numbers_revision.py

printf '\n\033[1mDone.\033[0m Outputs in %s\n' "$OUT"
