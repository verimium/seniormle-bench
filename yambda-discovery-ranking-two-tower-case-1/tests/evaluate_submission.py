#!/usr/bin/env python3
"""Validate and score a final Yambda discovery-ranking solution directory.

This module is installed in the separate Harbor verifier image under
``/tests``. It reads the explicitly transferred solver artifact from
``/app/solution``, the minimal public catalog snapshot under ``/tests/public``,
and evaluator-only encounter codes and hidden relevance labels under
``/tests/private``. Stage-independent mechanics come from the sibling
``ranking_evaluation.py`` module, which the builder installs from the same
source file as the agent-visible copy so the two cannot drift.

The solution directory must contain ``submission.parquet`` or
``submission.csv``. When both exist, parquet takes precedence. A submission is
structurally valid only when it has integer-valued ``uid``, ``item_id``, and
``rank`` columns; exactly 100 distinct catalog items and ranks 1 through 100 for
every target user; and no user-item pair encountered before the hidden-test
cutoff. Extra columns are ignored.

Valid rankings are scored with graded NDCG@10 over users represented in hidden
truth, then compared with the published quality threshold. Only a metric-passing
submission proceeds to a deterministic synthetic regression test of the
submitted model implementation. The repair verdict is a second acceptance gate,
while the measured ranking metric and dense reward remain intact when that gate
fails. Every solver outcome, including an invalid artifact, emits the canonical
result schema and exits zero. Nonzero exits are reserved for evaluator or
infrastructure failures.

Usage::

    python evaluate_submission.py <solution_dir>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd
from harbor_task import read_harbor_task
from ranking_evaluation import (
    RankingValidationError,
    ranked_items,
    read_ranking,
    score_rankings,
    validate_ranking,
)
from repair_validation import RepairValidationError, validate_recent_window_repair

TESTS = Path(__file__).resolve().parent
PUBLIC = TESTS / "public"
PRIVATE = TESTS / "private"
TASK_TOML = TESTS / "task.toml"
if not TASK_TOML.is_file():
    TASK_TOML = TESTS.parent / "task.toml"
TASK_CONTRACT = read_harbor_task(TASK_TOML)
TASK_ID = TASK_CONTRACT.task_id
THRESHOLDS = TASK_CONTRACT.thresholds
REPAIR_GATE = "recent_window_partition_repaired"
REPAIR_THRESHOLD = 1.0
REPAIR_TEST_UID = int(os.environ.get("REPAIR_TEST_UID", os.getuid()))
REPAIR_TEST_GID = int(os.environ.get("REPAIR_TEST_GID", os.getgid()))


class InvalidSubmission(ValueError):
    """Raised for solver-owned artifacts that cannot be scored."""


def fail(message: str) -> NoReturn:
    """Stop evaluation with a solver-owned invalid-submission outcome.

    Args:
        message: Human-readable reason the submission cannot be scored.

    Raises:
        InvalidSubmission: Always.
    """
    raise InvalidSubmission(message)


def invalid_result(message: str, *, metric_only: bool = False) -> dict[str, object]:
    """Build the canonical zero-reward result for an invalid solver artifact."""
    thresholds = dict(THRESHOLDS)
    if not metric_only:
        thresholds[REPAIR_GATE] = REPAIR_THRESHOLD
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "invalid",
        "reward": 0.0,
        "passed": False,
        "metrics": {},
        "thresholds": thresholds,
        "gates": {name: False for name in thresholds},
        "reason": message,
    }


def load_submission(solution_dir: Path) -> pd.DataFrame:
    """Read the supported submission file from ``solution_dir``.

    ``submission.parquet`` is preferred over ``submission.csv`` when both are
    present. File parsing errors are allowed to propagate to the caller.

    Args:
        solution_dir: Directory containing the solver's final artifact.

    Returns:
        The submission as loaded by pandas.

    Raises:
        InvalidSubmission: If the file is missing or cannot be parsed.
    """
    for name in ("submission.parquet", "submission.csv"):
        path = solution_dir / name
        if path.is_symlink():
            fail(f"{name} must be a regular file, not a symbolic link")
        if path.is_file():
            try:
                return read_ranking(path)
            except (OSError, TypeError, ValueError) as error:
                fail(f"could not read {name}: {type(error).__name__}: {error}")
    fail("missing submission.parquet or submission.csv")


def scored_result(
    metrics: dict[str, float | int],
    *,
    metric_only: bool,
    solution_dir: Path,
) -> dict[str, object]:
    """Apply the metric gate, then the submitted-code repair regression gate."""
    thresholds = dict(THRESHOLDS)
    gates = {
        name: bool(metrics[name] >= threshold) for name, threshold in thresholds.items()
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "task_id": TASK_ID,
        "status": "scored",
        "reward": float(metrics["quality_ndcg@10"]),
        "passed": all(gates.values()),
        "metrics": dict(metrics),
        "thresholds": thresholds,
        "gates": gates,
    }
    if metric_only:
        return result
    thresholds[REPAIR_GATE] = REPAIR_THRESHOLD
    result_metrics = dict(metrics)
    if not all(gates.values()):
        repair_value = 0.0
        acceptance_stage = "metric_gate_failed"
        reason = "repair regression runs only after all metric gates pass"
    else:
        try:
            validate_recent_window_repair(
                solution_dir,
                run_uid=REPAIR_TEST_UID,
                run_gid=REPAIR_TEST_GID,
            )
        except RepairValidationError as error:
            repair_value = 0.0
            acceptance_stage = "repair_regression_failed"
            reason = str(error)
        else:
            repair_value = 1.0
            acceptance_stage = "accepted"
            reason = None

    result_metrics[REPAIR_GATE] = repair_value
    gates[REPAIR_GATE] = repair_value >= REPAIR_THRESHOLD
    result.update(
        {
            "passed": all(gates.values()),
            "metrics": result_metrics,
            "thresholds": thresholds,
            "gates": gates,
            "acceptance_stage": acceptance_stage,
        }
    )
    if reason is not None:
        result["reason"] = reason
    return result


def validate_submission(submission: pd.DataFrame) -> dict[int, list[int]]:
    """Validate schema, coverage, ranking, catalog, and encounter constraints.

    Numeric values must be finite integers. The user set must exactly match
    ``target_users.parquet``; each user must have 100 unique items and every
    rank from 1 through 100 exactly once. Items must belong to the candidate
    catalog, and packed ``(uid << 32) | item_id`` values must not appear in the
    private pre-test encounter list.

    Args:
        submission: Raw submission frame. Extra columns are discarded from an
            internal copy; the caller's frame is not modified.

    Returns:
        A ``uid -> ranked item IDs`` mapping ordered by ascending rank.

    Raises:
        InvalidSubmission: If any validation rule fails.
    """
    targets = set(pd.read_parquet(PUBLIC / "data/target_users.parquet").uid.astype(int))
    try:
        submission = validate_ranking(submission, targets)
    except RankingValidationError as error:
        fail(str(error))

    catalog = set(
        pd.read_parquet(PUBLIC / "data/candidate_catalog.parquet").item_id.astype(int)
    )
    invalid_items = set(map(int, submission.item_id.unique())) - catalog
    if invalid_items:
        fail(
            f"submission contains {len(invalid_items)} items outside the "
            "candidate catalog"
        )

    codes = (
        submission.uid.to_numpy(dtype=np.uint64) << np.uint64(32)
    ) | submission.item_id.to_numpy(dtype=np.uint64)
    encountered = np.load(PRIVATE / "encounter_codes.npy")
    if np.isin(codes, encountered, assume_unique=False).any():
        fail(
            "submission recommends at least one item encountered before the "
            "hidden-test cutoff"
        )

    return ranked_items(submission)


def score(ranked: dict[int, list[int]]) -> dict[str, float | int]:
    """Score a validated ranking against evaluator-only hidden relevance.

    The private truth stores graded relevant items only. Per-user NDCG@10 is
    averaged over those users, matching the public evaluator's aggregation.

    Args:
        ranked: Ranked item IDs for every target user.

    Returns:
        The six-decimal ``quality_ndcg@10`` value and number of scored users.
    """
    raw = json.loads((PRIVATE / "truth.json").read_text(encoding="utf-8"))
    graded = {
        int(uid): {int(item): int(grade) for item, grade in rows}
        for uid, rows in raw["graded"].items()
    }
    return score_rankings(ranked, graded)


def main() -> int:
    """Run the private judge CLI and print an invalid or scored JSON report.

    Returns:
        Zero after every solver-owned outcome. CLI misuse and unhandled judge
        failures remain nonzero.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution_dir", type=Path)
    parser.add_argument(
        "--metric-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    solution_dir = args.solution_dir
    try:
        if not solution_dir.is_dir():
            fail(f"solution directory not found: {solution_dir}")
        metrics = score(validate_submission(load_submission(solution_dir)))
    except InvalidSubmission as error:
        print(
            json.dumps(
                invalid_result(str(error), metric_only=args.metric_only), indent=2
            )
        )
        return 0
    print(
        json.dumps(
            scored_result(
                metrics, metric_only=args.metric_only, solution_dir=solution_dir
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
