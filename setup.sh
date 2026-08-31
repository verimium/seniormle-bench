#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python3}
environments_dir="$repo_root/rl-environment/environments"
task_reader="$repo_root/rl-environment/scripts/harbor_task.py"

list_tasks() {
  "$python_bin" "$task_reader" "$environments_dir" --list-profiles
}

usage() {
  cat <<EOF
Usage: ./setup.sh <task-id> [builder arguments]
       ./setup.sh --list

Available task IDs:
$(list_tasks | sed 's/^/  /')
EOF
}

case ${1:-} in
--help|-h)
  usage
  exit 0
  ;;
--list)
  list_tasks
  exit 0
  ;;
"")
  usage >&2
  exit 2
  ;;
esac

task_id=$1
shift
profile=$(
  "$python_bin" "$task_reader" "$environments_dir" --find-profile "$task_id"
)
builder="$profile/build_task.py"
data_dir=${SENIORMLE_DATA_DIR:-"$repo_root/.data/yambda/50m"}
envs_dir=${SENIORMLE_ENVS_DIR:-"$repo_root/.venvs"}
tasks_dir=${SENIORMLE_TASKS_DIR:-"$repo_root/tasks"}

export PYTHONDONTWRITEBYTECODE=1
"$python_bin" "$builder" "$@" \
  --source-dir "$data_dir" \
  --env-dir "$envs_dir/$task_id" \
  --out "$tasks_dir/$task_id"

printf '\nReady Harbor task: %s\n' "$tasks_dir/$task_id"
