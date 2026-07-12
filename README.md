# SeniorMLE-Bench

A benchmark for long-horizon ML engineering agents.

A collection of tasks that place an agent inside a real production-shaped ML
Harbor environment: data, code, constraints, verifier, and a measurable outcome.
The score reflects whether the agent can ship working MLE tasks.

## What makes it different

**Real ML Tasks**
Production-shaped tasks and data workflows.

**Standard Environments**
[Harbor](https://www.harborframework.com/) environments with code, data, and constraints.

**Objective Verifier and Reward**
A verifier and reward signal tied to the measurable outcome.

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

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
