#!/usr/bin/env bash
# Evaluate every classifier run: per-run metrics, leaderboard, all pairwise DeLong and bootstrap
# comparisons. Usage: bash scripts/evaluate.sh CSV RUNS_DIR OUT
#   RUNS_DIR holds <exp>/<run>/model.pth (+ config.json, rnflt_norm_stats.json),
#   e.g. runs/checkpoints/glaucoma_classifier/ or the downloaded weights/classifiers/
set -eu
CSV="$1"
RUNS_DIR="$2"
OUT="$3"
python scripts/evaluate.py --csv "$CSV" --runs-dir "$RUNS_DIR" --output-dir "$OUT"
