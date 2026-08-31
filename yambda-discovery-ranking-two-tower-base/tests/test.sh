#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier
result_path=/logs/verifier/result.json
reward_path=/logs/verifier/reward.json
task_id=$(python /tests/harbor_task.py /tests/task.toml --field task_id)

python /tests/evaluate_submission.py /app/solution >"$result_path"
python /tests/verifier_result.py "$result_path" \
  --expect-task-id "$task_id" \
  --harbor-reward-output "$reward_path" \
  --quiet
