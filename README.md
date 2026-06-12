# SeniorMLE-Bench

A benchmark for long-horizon ML engineering agents.

Each task places a model inside a real production-shaped ML environment: data,
code, constraints, verifier, and a measurable outcome. The score reflects
whether the agent can ship working ML improvements, not whether it can answer
benchmark-style questions.

## What makes it different

**Real ML systems** — Tasks are drawn from recommendation, ranking, retrieval,
and data workflows.

**Runnable environments** — Each task ships with code, data, solution path, and
isolated verifier.

**Objective scoring** — Agents are judged by production-shaped metrics such as
ranking quality, correctness, and runtime constraints.

## Task layout

Every task is a self-contained directory:

```
<task-name>/
  instruction.md    # what the agent is asked to do
  task.toml         # task metadata
  environment/      # Dockerfile + requirements for the agent's workspace
  solution/         # reference solution and solve.sh
  tests/            # isolated verifier: scoring, metrics, test.sh
```

## Contact us

Send an email to [hello@verimium.com](mailto:hello@verimium.com).
