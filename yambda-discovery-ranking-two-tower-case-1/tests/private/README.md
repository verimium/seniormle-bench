# Intended Bug Summary: Anchor Capped Targets At The Stage Cutoff

## Defect

The case-study two-tower baseline inherits the clean base architecture and data
protocol, but its capped adjacent-target window is left-anchored. A stage cutoff
is the timestamp at which a ranking is produced; only interactions observed
before that timestamp belong to the fitting history. The per-user target cap is
different: it limits how many adjacent next-item targets from that observed
history contribute to model training.

When a user has more training targets than `max_train_targets_per_user`, the
partition keeps the oldest qualifying adjacent targets instead of the most
recent targets immediately before the final internal BCE validation suffix.
The bug therefore affects training supervision, not merely the number of raw
interactions read when encoding a user at inference time.

The defect is intentionally realistic: the validation split and target counts
look plausible, training still runs, candidate eligibility remains cutoff-safe,
and no private labels are read by the solver. The damage is semantic. The model
spends its limited training budget on stale behavior while recent preference
drift is underrepresented.

## Concrete Example

Suppose a user has 150 zero-indexed adjacent targets, two targets are reserved
for internal validation, and the training cap is four:

```text
Correct: training 144-147, validation 148-149
Planted: training   0-3, validation     4-5, targets 6-149 ignored
```

The correct window starts at `150 - 2 - 4 = 144`. The planted window starts at
zero. Users whose histories do not exceed the cap are unaffected.

## Invariant

For each user:

- `total_targets = len(sequence) - 1`
- `validation_count` is the final chronological BCE suffix.
- `history_training_count = total_targets - validation_count`
- `training_count = min(history_training_count, max_train_targets_per_user)`
- `window_start = history_training_count - training_count`

Therefore the selected window must satisfy:

```text
window_start + training_count + validation_count == total_targets
```

When the train cap binds, discarded targets must be a prefix of the user's
training history. The cap must never discard the most recent pre-validation or
pre-submission targets.

## Expected Repair

The semantic repair is one assignment in `chronological_target_partition`:

```python
window_starts = history_training - training
```

The repair must also restore the final-contiguous-suffix assertion above. That
assertion is part of the defect because the planted bounds-only check allows the
newest targets to be silently omitted. All other base-task behavior remains
unchanged: public files, candidate eligibility, explicit-state handling,
negative filtering, target weighting, architecture, and refit protocol.

## Prohibited Solver Shortcuts

- Do not inspect, tune against, or report hidden-test metrics.
- Do not read private labels, judge artifacts, or evaluator-owned hidden data
  from model-building code.
- Do not lower the acceptance threshold, change target users, change candidate
  eligibility, or alter relevance labels.
- Do not paper over the bug by increasing the target caps until the cap rarely
  binds. The lesson is about anchoring, not capacity.
- Do not replace the baseline with a different model family merely to avoid the
  partition invariant.

## Recorded Metrics And Threshold

These task-author records use seed 29 and the same data protocol for the planted
and repaired implementations. Public validation is the agent's development
signal; it is not a separate acceptance gate.

| Implementation | Public validation NDCG@10 | Hidden test NDCG@10 |
| --- | ---: | ---: |
| Planted current model | 0.008489 | 0.006341 |
| Repaired oracle | 0.019724 | 0.016592 |
| Acceptance threshold | n/a | 0.016592 |

The hidden acceptance threshold is exactly the repaired oracle value. A
submission must match or exceed that score; no additional calibration margin is
applied.

## Automated Acceptance

The verifier applies three checks in order:

1. Validate the ranking schema, user coverage, ranks, item uniqueness, catalog
   membership, and cutoff-correct candidate eligibility.
2. Score hidden-test NDCG@10 and require at least `0.016592`.
3. Import `/app/solution/train_current_model.py` in an isolated subprocess and
   call `chronological_target_partition` on these synthetic scenarios:

| Sequence lengths | Validation fraction | Training cap | Validation cap |
| --- | ---: | ---: | ---: |
| `2, 6, 151, 1001` | 0.10 | 4 | 2 |
| `1, 3, 8, 33, 129` | 0.25 | 7 | 3 |
| `1, 2, 4, 10, 65` | 0.00 | 2 | 0 |

The regression compares `window_starts`, `training_counts`,
`validation_counts`, and `history_training_counts` with independently computed
right-anchored results. It covers binding and nonbinding caps, short histories,
multiple validation fractions, and a zero-validation case. Missing code, a
missing helper, import failure, timeout, or any array mismatch fails this gate.
The test does not inspect source text. Prose is not an acceptance gate, and the
verifier has no network or LLM dependency.

## Private Reference Solution

The repaired implementation is shared with the base task rather than duplicated
in this case profile. Its canonical authoring source is
`../../../task_source/yambda_two_tower/assets/reference_solver.py`.

During task generation, the builder materializes the following evaluator-only
artifacts:

- `/tests/private/reference_solution/reference_solver.py`: repaired
  training/ranking CLI with the right-anchored target partition restored;
- `/tests/private/reference_solution/two_tower_model.py`: unchanged model
  primitives; and
- `/tests/private/test_recent_window_partition.py`: synthetic positive and
  negative controls for the intended repair.

These files exist for task review and calibration. They are not solver-visible,
an ensemble, or tuned against hidden metrics.

## Authoring Regression Test

`/tests/private/test_recent_window_partition.py` is an authoring check, not the
per-submission gate described above. It runs a binding-cap example against the
repaired private reference and asserts both the right-anchor formula and the
final-contiguous-suffix invariant. It then runs the same example against the
planted model and requires that model to fail as a negative control.
