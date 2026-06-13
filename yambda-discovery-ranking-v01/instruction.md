# Build a materially better music-discovery ranker

Build the strongest ranking model you can for the "Made for You" shelf of a
music streaming service. Per-user candidate eligibility, cutoff-specific
explicit preference state, and a temporal backtest are provided. Candidate
eligibility is platform-owned; all modeling within that candidate boundary is
in scope.

## Data

All inputs are under `/task/data/`. Training data ends at timestamp `24786800`.
A 30-minute gap precedes a seven-day public validation window. The hidden test
is the following seven-day window after another 30-minute gap.

For each event family there is a `train_*.parquet` and `val_*.parquet` file:

| Family | Meaning | Columns |
| --- | --- | --- |
| `listens` | Playback events | `uid`, `timestamp`, `item_id`, `is_organic`, `played_ratio_pct`, `track_length_seconds` |
| `likes` | Explicit positive actions | `uid`, `timestamp`, `item_id`, `is_organic` |
| `dislikes` | Explicit negative actions | `uid`, `timestamp`, `item_id`, `is_organic` |
| `unlikes` | Reversal of a prior like | `uid`, `timestamp`, `item_id`, `is_organic` |
| `undislikes` | Reversal of a prior dislike | `uid`, `timestamp`, `item_id`, `is_organic` |

`is_organic=1` means the listener sought out the track; `0` means the action
came through a recommendation surface. A listen is a conventional positive at
50% completion. The discovery-quality objective below uses the stronger 80%
threshold.

The two public windows have different roles:

1. Build and tune a validation ranker using only `train_*` as history and
   `val_*` as subsequent outcomes.
2. After selecting the approach, refit it on `train_* + val_*`. At hidden-test
   time the validation events are observed history, not recommendable outcomes.

Do not use `val_*` events as features when measuring a validation ranking. They
may and should be used when refitting the final hidden-test ranker.

Additional files under `/task/data/`:

- `candidate_catalog.parquet`: the 30,000 retrieval-catalog tracks, ordered by
  training positive-listen popularity.
- `target_users.parquet`: users with at least 20 positive listens in the 90 days
  before the training cutoff, selected without future labels.
- `validation_preference_state.parquet`: active explicit state at the training
  cutoff, for validation ranking.
- `test_preference_state.parquet`: active explicit state after validation, for
  the final hidden-test ranking. State is `liked` or `disliked`; a pair may have
  both active channels.
- `validation_eligible_candidates.npz` and `test_eligible_candidates.npz`:
  cutoff-correct candidate sets for every target user. Each contains aligned
  arrays `uids`, `item_ids`, and `packed_eligible`. For user row `r`, recover
  the Boolean item mask with `np.unpackbits(packed_eligible[r],
  count=len(item_ids), bitorder="little")`.

The exact available Python distribution inventory is listed in
`/task/requirements.txt`. Future test interactions are hidden and used only by
the separate Harbor verifier.

## Discovery objective

Rank only the candidates returned for the relevant cutoff. Relevance is based
on organic behavior in the immediately following window:

- grade 3: the user organically likes the track;
- grade 1: the user organically listens to at least 80% of the track;
- grade 0: otherwise.

If both positive events occur, grade 3 wins. Ranking quality is measured with
graded NDCG@10. DCG uses gain `2^grade - 1` and discount
`1/log2(rank + 1)`, then averages per-user NDCG over users with at least one
relevant item.

Use this published gate for the public validation backtest. The final hidden
test submission must also pass it:

| Gate | Required score |
| --- | ---: |
| quality NDCG@10 | 0.025 |

The Yambda logs contain interactions, not impression propensities. Treat the
organic objective as a robust offline target, not as a causal estimate of
online lift.

## Public tools

`/task/evaluate_public.py` scores a validation-stage ranking against the public
window. Pass the path to a ranking with the same schema as the final
deliverable:

```sh
python /task/evaluate_public.py solution/validation.parquet
```

Stage-independent evaluation helpers are available in
`/task/ranking_evaluation.py`. The builder installs the same checked-in module
in the public agent image and private verifier image so scoring mechanics cannot
drift. The public copy contains no hidden labels, evaluator-owned data paths, or
pass thresholds.

The public evaluator uses `validation_eligible_candidates.npz`; it is not a
validator for the final artifact. No model, fitted representation, example
submission, or modeling architecture is supplied. Build and tune the best model
you can, then refit the selected approach as described above.

## Deliverable

Write `/app/solution/submission.parquet` or
`/app/solution/submission.csv`, with exactly 100 rows for every target user and
columns `uid`, `item_id`, `rank`. Ranks must be the integers 1-100 with no ties
or duplicate items. Every submitted item must be present in that user's
`test_eligible_candidates.npz` candidate set.

Leave the code that builds the artifact and a concise
`/app/solution/README.md` in `/app/solution/`. Keep the best valid submission
saved throughout the attempt so it survives the deadline.

## Constraints

- Write deliverables and intermediate artifacts only under `/app/solution/`.
- Treat `/task/` and the installed Python environment as read-only.
- Do not invoke nested agents or attempt to access verifier files, credentials,
  other trials, or host paths.
- Do not install packages or use network access except for the model provider
  connection managed by Harbor.
- You have up to 30 minutes. Improve the model as much as possible within that
  limit. Structural validity alone is not sufficient; the final submission
  should pass the published quality gate.
