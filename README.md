# SeniorMLE-Bench

A benchmark for long-horizon machine-learning engineering agents.

This repository intentionally contains task code only. Dataset files and built
Harbor tasks are not stored in Git. `setup.sh` downloads the pinned Yambda
source files directly from their origin, verifies their SHA-256 checksums, and
materializes a complete task under `tasks/`.

The source tree includes evaluator-only build inputs. Give Harbor the generated
`tasks/<task-id>` directory; do not mount this repository itself into the agent
environment.

## Prerequisites

- Python 3
- [`uv`](https://docs.astral.sh/uv/)
- Docker
- Harbor 0.22.0 for task execution

## Build a task

List the available task IDs:

```sh
./setup.sh --list
```

Build one task:

```sh
./setup.sh yambda-discovery-ranking-two-tower-base
```

The first build downloads the source dataset and creates a pinned Python 3.12
authoring environment. Later builds reuse both. The resulting Harbor task is
written to:

```text
tasks/yambda-discovery-ranking-two-tower-base/
```

Run it with Harbor after setup completes:

```sh
harbor run \
  --path tasks/yambda-discovery-ranking-two-tower-base \
  --agent codex \
  --model gpt-5.6-sol
```

Additional arguments after the task ID are passed to its builder. For example,
`--force-download` replaces the cached upstream files. `SENIORMLE_DATA_DIR`,
`SENIORMLE_ENVS_DIR`, and `SENIORMLE_TASKS_DIR` can relocate the generated
directories.

## Repository layout

```text
setup.sh                         # task bootstrap entrypoint
source-manifest.json             # hashes for every distributed source file
rl-environment/environments/     # deterministic task builders
rl-environment/scripts/          # shared contracts and validators
tasks/                           # generated locally; ignored by Git
```

Run the code-only distribution check with:

```sh
python3 scripts/validate_distribution.py .
```
