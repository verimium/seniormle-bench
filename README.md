# SeniorMLE-Bench

A benchmark for long-horizon machine-learning engineering agents.

Each top-level directory is a complete Harbor task except for generated data.
The task code, agent environment, verifier, oracle solution, and Harbor contract
are stored in Git. Dataset-derived files are intentionally excluded.

Harbor does not run setup scripts automatically. Before running a task, execute
that task's `setup.sh`. It downloads the pinned Yambda source files directly
from their origin, verifies their SHA-256 checksums, builds the derived inputs,
and hydrates the same top-level task directory.

## Prerequisites

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Docker
- Harbor 0.22.0 for task execution

## Prepare and run a task

```sh
./yambda-discovery-ranking-two-tower-base/setup.sh

harbor run \
  --path yambda-discovery-ranking-two-tower-base \
  --agent codex \
  --model gpt-5.6-sol
```

The first setup creates shared `.data/` and `.venvs/` caches at the repository
root. Later task setups reuse them. Both the caches and hydrated task data are
ignored by Git.

Builder options may be passed to the task setup. For example:

```sh
./yambda-discovery-ranking-v01/setup.sh --force-download
```

`SENIORMLE_DATA_DIR` and `SENIORMLE_ENVS_DIR` can relocate the shared caches.

## Repository layout

```text
yambda-discovery-ranking-v01/                 # data-free Harbor task
  task.toml
  instruction.md
  environment/
  solution/
  tests/
  setup.sh                                    # hydrates this task in place
yambda-discovery-ranking-two-tower-base/      # another top-level task
yambda-discovery-ranking-two-tower-case-1/    # another top-level task
source-manifest.json                          # exported-source hashes
scripts/validate_distribution.py              # distribution validation
```

Validate a checkout without downloading data:

```sh
python3 scripts/validate_distribution.py .
```

The private `mmoghimi/pitch` repository is the source of truth. Its sync
workflow deterministically regenerates this repository and opens a pull request
when task code changes.
