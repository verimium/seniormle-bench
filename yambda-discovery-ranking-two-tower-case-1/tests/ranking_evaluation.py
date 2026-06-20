"""Shared public primitives for Yambda discovery-ranking evaluation.

Both the solver-visible validation evaluator and the private submission judge
import this module from the generated task's ``public`` directory. It owns the
stage-independent contract: reading a ranking, validating its shape, converting
it to per-user rank order, looking up packed candidate eligibility, and
computing graded NDCG@10.

The module deliberately contains no event-derived labels, private artifact
paths, or pass thresholds. Each evaluator supplies its own relevance data and
per-cutoff eligibility checks around these common operations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


__all__ = [
    "CandidateSet",
    "QUALITY_K",
    "RankingValidationError",
    "REQUIRED_COLUMNS",
    "TOP_N",
    "ndcg_at_k",
    "ranked_items",
    "read_ranking",
    "score_rankings",
    "validate_ranking",
]

TOP_N = 100
QUALITY_K = 10
REQUIRED_COLUMNS = ("uid", "item_id", "rank")


class RankingValidationError(ValueError):
    """Report a ranking-schema or structural-contract violation."""


class CandidateSet:
    """Provide membership lookups over one packed candidate-set archive.

    The archive contains aligned ``uids``, ``item_ids``, and
    ``packed_eligible`` arrays. Candidate column ``c`` is represented by bit
    ``c % 8`` of byte ``c // 8`` using little-endian bit order. Arrays are
    copied into memory so they remain valid after the archive is closed.
    """

    def __init__(self, path: Path):
        """Load an eligibility archive and build ID-to-position indexes.

        Args:
            path: Path to a ``.npz`` archive with the candidate-set schema.
        """
        with np.load(path, allow_pickle=False) as artifact:
            self.uids = artifact["uids"].astype(np.int64, copy=True)
            self.item_ids = artifact["item_ids"].astype(np.int64, copy=True)
            self.packed = artifact["packed_eligible"].astype(np.uint8, copy=True)
        self.uid_row = {int(uid): row for row, uid in enumerate(self.uids)}
        self.item_column = {
            int(item): column for column, item in enumerate(self.item_ids)
        }

    def is_eligible(self, uid: int, item_id: int) -> bool:
        """Return whether ``item_id`` is eligible for ``uid``.

        IDs absent from the target-user or candidate-item arrays are treated as
        ineligible rather than raising an exception.

        Args:
            uid: Target user ID to query.
            item_id: Candidate item ID to query.

        Returns:
            ``True`` only when the corresponding packed eligibility bit is set.
        """
        row = self.uid_row.get(int(uid))
        column = self.item_column.get(int(item_id))
        if row is None or column is None:
            return False
        byte, bit = divmod(column, 8)
        return bool((int(self.packed[row, byte]) >> bit) & 1)


def read_ranking(path: Path) -> pd.DataFrame:
    """Read a parquet or CSV ranking into a pandas frame.

    Args:
        path: Input path. A ``.parquet`` suffix selects parquet; every other
            suffix is interpreted as CSV.

    Returns:
        The ranking frame as loaded by pandas.
    """
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def validate_ranking(
    frame: pd.DataFrame,
    targets: set[int],
) -> pd.DataFrame:
    """Normalize and validate the stage-independent ranking contract.

    Required values must be finite integers. The user set must exactly match
    ``targets``; every user must have 100 rows, distinct items, and each rank
    from 1 through 100 exactly once. Cutoff-specific candidate eligibility is
    intentionally left to the calling evaluator.

    Args:
        frame: Raw ranking frame. Extra columns are discarded from an internal
            copy; the caller's frame is not modified.
        targets: Exact set of target user IDs the ranking must cover.

    Returns:
        A normalized frame containing only the required ``int64`` columns.

    Raises:
        RankingValidationError: If the ranking violates the shared contract.
    """
    required = list(REQUIRED_COLUMNS)
    if not all(column in frame for column in required):
        raise RankingValidationError(f"submission requires columns {required}")

    normalized = frame[required].copy()
    for column in required:
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        values = numeric.to_numpy(dtype=float)
        if numeric.isna().any() or not np.isfinite(values).all():
            raise RankingValidationError(
                f"submission {column} must be finite numeric values"
            )
        if not np.equal(numeric, np.floor(numeric)).all():
            raise RankingValidationError(
                f"submission {column} must contain integers"
            )
        normalized[column] = numeric.astype(np.int64)

    submitted_users = set(map(int, normalized.uid.unique()))
    if submitted_users != targets:
        raise RankingValidationError(
            f"submission user set mismatch: missing={len(targets - submitted_users)} "
            f"extra={len(submitted_users - targets)}"
        )
    counts = normalized.groupby("uid").size()
    if not (counts == TOP_N).all():
        raise RankingValidationError(
            f"submission must contain exactly {TOP_N} rows for every target user"
        )
    if normalized.duplicated(["uid", "item_id"]).any():
        raise RankingValidationError(
            "submission contains duplicate (uid, item_id) pairs"
        )
    valid_ranks = set(range(1, TOP_N + 1))
    ranks_are_valid = normalized.groupby("uid")["rank"].apply(
        lambda values: set(map(int, values)) == valid_ranks
    )
    if not ranks_are_valid.all():
        raise RankingValidationError(
            f"each user must have every rank from 1 through {TOP_N} exactly once"
        )
    return normalized


def ranked_items(frame: pd.DataFrame) -> dict[int, list[int]]:
    """Convert a validated ranking frame to ordered items keyed by user.

    Args:
        frame: Frame that has passed :func:`validate_ranking`.

    Returns:
        A ``uid -> item IDs`` mapping ordered by ascending rank.
    """
    ordered = frame.sort_values(["uid", "rank"])
    return {
        int(uid): list(map(int, values))
        for uid, values in ordered.groupby("uid").item_id
    }


def ndcg_at_k(ranked: list[int], gold: dict[int, int], k: int) -> float:
    """Compute graded normalized discounted cumulative gain at ``k``.

    Relevance gain is ``2**grade - 1`` and the discount at one-based rank
    ``r`` is ``1 / log2(r + 1)``. Items absent from ``gold`` have grade zero.

    Args:
        ranked: Item IDs in predicted rank order.
        gold: Mapping from relevant item IDs to integer relevance grades.
        k: Maximum number of predicted positions to score.

    Returns:
        For a valid duplicate-free ranking, NDCG in ``[0, 1]``; returns
        ``0.0`` when the ideal gain is zero.
    """
    gains = np.asarray([gold.get(item, 0) for item in ranked[:k]], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum((np.power(2.0, gains) - 1.0) * discounts))
    ideal = np.asarray(sorted(gold.values(), reverse=True)[:k], dtype=np.float64)
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float(np.sum((np.power(2.0, ideal) - 1.0) * ideal_discounts))
    return dcg / idcg if idcg else 0.0


def score_rankings(
    ranked: dict[int, list[int]],
    graded: dict[int, dict[int, int]],
) -> dict[str, float | int]:
    """Aggregate per-user NDCG@10 over users with relevance labels.

    Args:
        ranked: Ranked item IDs for every target user.
        graded: Relevant item grades keyed by user and item.

    Returns:
        The six-decimal ``quality_ndcg@10`` value and number of scored users.
    """
    quality = float(
        np.mean(
            [
                ndcg_at_k(ranked[uid], gold, QUALITY_K)
                for uid, gold in graded.items()
            ]
        )
    )
    return {
        "quality_ndcg@10": round(quality, 6),
        "quality_users": len(graded),
    }
