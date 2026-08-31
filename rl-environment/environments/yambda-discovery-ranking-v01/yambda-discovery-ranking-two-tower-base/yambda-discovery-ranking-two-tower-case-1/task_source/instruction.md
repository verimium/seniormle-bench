# Improve a production music-discovery two-tower

You have joined the recommendations team for a music streaming service. The
"Made for You" shelf is served by a mature production two-tower model. Its
chronological data protocol and indexed serving path are sound, but its offline
retrieval quality no longer clears the team's next launch gate.

Your job is to investigate the production model, improve its retrieval quality,
and ship a better ranking artifact. Use the implementation and public validation
data to form and test your own diagnosis.

The source used to reproduce the production model is included under
`/task/current_model/`. You may copy it into `/app/solution/`, edit it, add
tests, and improve its training objective, but keep the data protocol,
two-tower serving contract, and evaluation boundary intact.

## Production Context

The model must satisfy a strict request-time latency budget. Item vectors are
precomputed offline. At request time, one user vector is computed from observed
interaction history and compared with candidate item vectors by a batchable
two-tower score. Do not introduce a cross-encoder, per-candidate sequence model,
or user-ID lookup table.

The current trainer is a mature baseline with chronological internal
validation. Focus your investigation on how its training objective selects and
uses negative examples. Diagnose limitations in the current negative-sampling
strategy and improve it under the fixed CPU budget. Validate changes with the
public backtest. The final approach must still train and rank within the supplied
CPU environment while preserving the serving and data boundaries above.

## What You Have

All public inputs are under `/task/data/`. A stage cutoff is the timestamp at
which observed history and candidate eligibility are frozen for a ranking. The
public validation stage cutoff is `24786800`, when `train_*` ends. A 30-minute
gap precedes the seven-day `val_*` window. For the final submission, that
validation window becomes observed history, and another 30-minute gap precedes
the hidden seven-day outcome window.

The package includes:

- `/task/current_model/train_current_model.py`: the production training and
  ranking script. It can reproduce the public validation ranking and produce a
  hidden-test submission after refitting.
- `/task/current_model/two_tower_model.py`: model primitives used by the
  production script.
- `/task/evaluate_public.py`: the public validation evaluator for development.
- `/task/data/validation_eligible_candidates.npz` and
  `/task/data/test_eligible_candidates.npz`: platform-owned candidate sets for
  the validation and hidden-test cutoffs.
- `/task/data/validation_preference_state.parquet` and
  `/task/data/test_preference_state.parquet`: active explicit preference state
  at the corresponding cutoffs.
- `/task/requirements.txt`: the exact offline Python distribution inventory.

You can reproduce the production public backtest with:

```sh
python /task/current_model/train_current_model.py \
  --stage validation \
  --public-dir /task \
  --output /app/solution/production_validation.parquet
python /task/evaluate_public.py /app/solution/production_validation.parquet
```

## Data

For each event family there is a `train_*.parquet` and `val_*.parquet` file:

| Family | Meaning | Columns |
| --- | --- | --- |
| `listens` | Playback events | `uid`, `timestamp`, `item_id`, `is_organic`, `played_ratio_pct`, `track_length_seconds` |
| `likes` | Explicit positive actions | `uid`, `timestamp`, `item_id`, `is_organic` |
| `dislikes` | Explicit negative actions | `uid`, `timestamp`, `item_id`, `is_organic` |
| `unlikes` | Reversal of a prior like | `uid`, `timestamp`, `item_id`, `is_organic` |
| `undislikes` | Reversal of a prior dislike | `uid`, `timestamp`, `item_id`, `is_organic` |

`is_organic=1` means the listener sought out the track; `0` means the action came
through a recommendation surface. A listen is a conventional positive at 50%
completion. The discovery-quality objective below uses the stronger 80%
threshold.

The two public windows have different roles:

1. For model development and selection, train only from `train_*` and score the
   subsequent `val_*` outcomes.
2. After selecting an improvement, refit it on `train_* + val_*`. At hidden-test
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

Validation candidates exclude all `train_*` encounters. Test candidates exclude
all `train_*` and `val_*` encounters. Future test interactions are hidden and
used only by the separate Harbor verifier.

## Discovery Objective

Rank only the candidates returned for the relevant cutoff. Relevance is based
on organic behavior in the immediately following window:

- grade 3: the user organically likes the track;
- grade 1: the user organically listens to at least 80% of the track;
- grade 0: otherwise.

If both positive events occur, grade 3 wins. Ranking quality is measured with
graded NDCG@10. DCG uses gain `2^grade - 1` and discount
`1/log2(rank + 1)`, then averages per-user NDCG over users with at least one
relevant item.

The public evaluator reports this same objective for development, but its score
is not a separate acceptance stage. Final acceptance is determined on the
hidden test using the same objective. The Yambda logs contain interactions, not
impression propensities; treat the organic objective as a robust offline target,
not as a causal estimate of online lift.

## Deliverable

Submit one final ranking as `/app/solution/submission.parquet` or
`/app/solution/submission.csv`. It must be a regular file, not a symbolic link.
If both files exist, the verifier evaluates `submission.parquet` and ignores
`submission.csv`.

The ranking must:

- include the columns `uid`, `item_id`, and `rank`; only these columns are
  evaluated;
- use finite, integer-valued entries in all three required columns;
- cover exactly the users in `target_users.parquet`, with no missing or extra
  users;
- contain exactly 100 rows and 100 distinct items for every target user;
- contain every integer rank from 1 through 100 exactly once per user; and
- use only items present in that user's `test_eligible_candidates.npz`
  candidate set.

Also leave the improved implementation used to build the artifact at
`/app/solution/train_current_model.py`, along with any local modules it imports.
A concise `/app/solution/README.md` with the diagnosis, before-and-after public
score, model change, and submission command is useful for the experiment record,
but it is not an automated acceptance gate.

## Constraints

- Write deliverables and intermediate artifacts only under `/app/solution/`.
- Treat `/task/` and the installed Python environment as read-only.
- Do not invoke nested agents or attempt to access verifier files, credentials,
  other trials, or host paths.
- Do not install packages or use network access except for the model-provider
  connection managed by Harbor.
- Do not alter cutoff definitions, target users, candidate eligibility, or
  relevance labels.
- Improve the production pipeline in place. Do not substitute an unrelated
  model family or evade the indexed two-tower serving constraint.
- The environment includes CPU PyTorch 2.8.0. No accelerator is available.
- You have up to 60 minutes. Prioritize measured validation improvements, a
  correct final refit, and a valid submission over broad architecture search.
- Keep the best valid submission saved throughout the attempt so it survives
  the deadline.
