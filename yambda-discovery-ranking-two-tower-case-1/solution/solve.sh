#!/usr/bin/env bash
set -euo pipefail

mkdir -p /app/solution
cp /solution/reference_solver.py /app/solution/train_current_model.py
cp /solution/two_tower_model.py /app/solution/two_tower_model.py
python /app/solution/train_current_model.py \
  --stage submission \
  --public-dir /task \
  --output /app/solution/submission.parquet

cat > /app/solution/README.md <<'EOF'
# Repaired recent-window two-tower solution

The deployed model capped each user's adjacent training targets but anchored
that cap at the beginning of the sequence, retaining stale targets for long
histories. This reference repairs `chronological_target_partition` by setting
`window_starts = history_training - training`, so every capped training window
is the most recent suffix before validation or submission. The same corrected
partition is used for validation and the final train-plus-validation refit; the
model architecture, loss, candidate eligibility, and evaluation boundaries are
otherwise unchanged.
EOF
