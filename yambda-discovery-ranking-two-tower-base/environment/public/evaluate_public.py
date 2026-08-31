#!/usr/bin/env python3
"""Score a validation-stage ranking against public discovery labels.

``build_task.py`` publishes this module with the solver-visible task files. It
loads target users, validation eligibility, and validation-window events from
the adjacent ``data`` directory, then evaluates one ranking supplied on the
command line. Stage-independent mechanics come from the adjacent public
``ranking_evaluation.py`` module. The builder installs the same checked-in
module in the private judge, so public and private scoring mechanics cannot
drift. A relevant candidate receives grade 3 for an organic like or grade 1 for
an organic listen with at least 80 percent completion; a like takes precedence
when both events exist.

Before scoring, the ranking must contain exactly 100 distinct eligible items
for every target user, with integer-valued ``uid``, ``item_id``, and ``rank``
columns and ranks 1 through 100 exactly once per user. Extra columns are
ignored. Quality is graded NDCG@10, averaged only over users who have at least
one eligible relevant item in the validation window.

The evaluator prints a JSON object containing the model path and scores. It
does not apply the task's pass threshold; its purpose is model development and
selection. The private judge applies the threshold to the final submission.

Usage::

    python evaluate_public.py <validation-ranking.parquet|csv>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from ranking_evaluation import (
    CandidateSet,
    RankingValidationError,
    ranked_items,
    read_ranking,
    score_rankings,
    validate_ranking,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LISTEN_STRONG = 80


def validation_truth(
    eligibility: CandidateSet, targets: set[int]
) -> dict[int, dict[int, int]]:
    """Build cutoff-correct graded relevance from public validation events.

    Only target-user events for eligible candidates are retained. Organic
    listens completed to at least 80 percent receive grade 1, and organic likes
    receive grade 3. Repeated user-item events are collapsed, and processing
    likes second ensures that grade 3 wins over grade 1.

    Args:
        eligibility: Validation-cutoff candidate membership service.
        targets: Complete set of target user IDs.

    Returns:
        A nested ``uid -> item_id -> grade`` mapping. Users without an eligible
        relevant event are omitted.
    """
    listens = pd.read_parquet(DATA / "val_listens.parquet")
    likes = pd.read_parquet(DATA / "val_likes.parquet")
    strong = listens[
        listens.uid.isin(targets)
        & (listens.is_organic == 1)
        & (listens.played_ratio_pct >= LISTEN_STRONG)
    ][["uid", "item_id"]].drop_duplicates()
    organic_likes = likes[
        likes.uid.isin(targets) & (likes.is_organic == 1)
    ][["uid", "item_id"]].drop_duplicates()

    graded: dict[int, dict[int, int]] = {}
    for raw_uid, raw_item in strong.itertuples(index=False, name=None):
        uid, item = int(raw_uid), int(raw_item)
        if eligibility.is_eligible(uid, item):
            graded.setdefault(uid, {})[item] = 1
    for raw_uid, raw_item in organic_likes.itertuples(index=False, name=None):
        uid, item = int(raw_uid), int(raw_item)
        if eligibility.is_eligible(uid, item):
            graded.setdefault(uid, {})[item] = 3
    return graded


def load_submission(
    path: Path, eligibility: CandidateSet, targets: set[int]
) -> dict[int, list[int]]:
    """Load and validate a public validation ranking.

    Files ending in ``.parquet`` are read as parquet; every other suffix is
    read as CSV. The required columns must be integer-valued, cover exactly the
    target-user set, contain 100 rows and ranks 1 through 100 per user, avoid
    duplicate user-item pairs, and contain only cutoff-eligible items.

    Args:
        path: Ranking file to read.
        eligibility: Validation-cutoff candidate membership service.
        targets: Exact set of users the ranking must cover.

    Returns:
        A ``uid -> ranked item IDs`` mapping ordered by ascending rank.

    Raises:
        ValueError: If the ranking violates the required schema or invariants.
    """
    frame = validate_ranking(read_ranking(path), targets)
    if not all(
        eligibility.is_eligible(uid, item)
        for uid, item in frame[["uid", "item_id"]].itertuples(index=False, name=None)
    ):
        raise RankingValidationError(
            "submission contains an item rejected by eligibility service"
        )
    return ranked_items(frame)


def score(
    ranked: dict[int, list[int]],
    graded: dict[int, dict[int, int]],
) -> dict[str, float | int]:
    """Aggregate per-user NDCG@10 over users with public relevance labels.

    Args:
        ranked: Ranked item IDs for every target user.
        graded: Eligible validation relevance keyed by user and item.

    Returns:
        The six-decimal ``quality_ndcg@10`` value and number of scored users.
    """
    return score_rankings(ranked, graded)


def main() -> None:
    """Run the public evaluator CLI and print its JSON score report."""
    targets_frame = pd.read_parquet(DATA / "target_users.parquet")
    targets = set(targets_frame.uid.astype(int))
    eligibility = CandidateSet(DATA / "validation_eligible_candidates.npz")
    graded = validation_truth(eligibility, targets)

    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: evaluate_public.py <validation-ranking.parquet|csv>"
        )
    path = Path(sys.argv[1])
    ranked = load_submission(path, eligibility, targets)
    print(json.dumps({"model": str(path), "scores": score(ranked, graded)}, indent=2))


if __name__ == "__main__":
    main()
