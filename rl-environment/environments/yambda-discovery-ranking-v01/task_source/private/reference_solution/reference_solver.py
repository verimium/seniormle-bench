#!/usr/bin/env python3
"""Build the private Yambda discovery-ranking reference submission.

The model is cutoff-correct and supports both stages:

* ``validation`` fits on ``train_*`` history and ranks validation candidates.
* ``submission`` fits on ``train_* + val_*`` history and ranks hidden-test
  candidates.

The solver is label-blind. Its only task input is the supplied Harbor agent
package at ``/task``; scoring is performed by the separate verifier image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DAY = 86_400
TRAIN_END = 24_786_800
VALIDATION_END = 25_393_400
LISTEN_POSITIVE = 50
RECENT_DAYS = 21
NEIGHBOURS = 120
PROFILE_DEPTH = 80
TOP_N = 100
ORGANIC_WEIGHT = 2.0
ACTIVE_LIKE_WEIGHT = 4.0


class CandidateSet:
    """Read one packed per-user candidate-set data file."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as artifact:
            self.uids = artifact["uids"].astype(np.int64, copy=True)
            self.item_ids = artifact["item_ids"].astype(np.int64, copy=True)
            self.packed = artifact["packed_eligible"].astype(np.uint8, copy=True)
        self.uid_row = {int(uid): row for row, uid in enumerate(self.uids)}
        self.item_column = {
            int(item): column for column, item in enumerate(self.item_ids)
        }

    def candidates_for(self, uid: int) -> np.ndarray:
        row = self.uid_row[int(uid)]
        mask = np.unpackbits(
            self.packed[row], count=len(self.item_ids), bitorder="little"
        ).astype(bool, copy=False)
        return self.item_ids[mask].copy()

    def filter_ranked(self, uid: int, ranked_items: list[int]) -> np.ndarray:
        row = self.uid_row[int(uid)]
        mask = np.unpackbits(
            self.packed[row], count=len(self.item_ids), bitorder="little"
        ).astype(bool, copy=False)
        selected: list[int] = []
        selected_set: set[int] = set()
        for raw_item in ranked_items:
            item = int(raw_item)
            column = self.item_column.get(item)
            if column is None or not mask[column] or item in selected_set:
                continue
            selected.append(item)
            selected_set.add(item)
        return np.asarray(selected, dtype=np.int64)


def fit_similarity(
    similarity_events: pd.DataFrame,
    active_likes: pd.DataFrame,
    candidate_items: np.ndarray,
    history_end: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Fit reference similarity with active likes injected at weight four."""
    cutoff = history_end - RECENT_DAYS * DAY
    recent = similarity_events[
        (similarity_events.timestamp >= cutoff)
        & similarity_events.item_id.isin(candidate_items)
    ].copy()
    items = np.asarray(candidate_items, dtype=np.int64)
    users = np.sort(recent.uid.unique())
    item_index = {int(value): index for index, value in enumerate(items)}
    user_index = {int(value): index for index, value in enumerate(users)}
    rows = recent.uid.map(user_index).to_numpy()
    cols = recent.item_id.map(item_index).to_numpy()
    values = np.where(recent.is_organic.to_numpy() == 1, ORGANIC_WEIGHT, 1.0).astype(
        np.float32
    )

    relevant_likes = active_likes[
        active_likes.uid.isin(users) & active_likes.item_id.isin(items)
    ]
    rows = np.concatenate([rows, relevant_likes.uid.map(user_index).to_numpy()])
    cols = np.concatenate([cols, relevant_likes.item_id.map(item_index).to_numpy()])
    values = np.concatenate(
        [values, np.full(len(relevant_likes), ACTIVE_LIKE_WEIGHT, dtype=np.float32)]
    )

    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))
    matrix.sum_duplicates()
    matrix.data = np.log1p(matrix.data)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=0)).A1
    norms[norms == 0] = 1.0
    normalised = matrix.multiply(sparse.csr_matrix(1.0 / norms)).tocsr()
    similarity = (normalised.T @ normalised).tocsr()

    data_out: list[np.ndarray] = []
    indices_out: list[np.ndarray] = []
    indptr = [0]
    for row_id in range(similarity.shape[0]):
        start, end = similarity.indptr[row_id], similarity.indptr[row_id + 1]
        data = similarity.data[start:end]
        indices = similarity.indices[start:end]
        if len(data) > NEIGHBOURS:
            selected = np.argpartition(-data, NEIGHBOURS)[:NEIGHBOURS]
            data, indices = data[selected], indices[selected]
        data_out.append(data)
        indices_out.append(indices)
        indptr.append(indptr[-1] + len(data))
    return (
        sparse.csr_matrix(
            (
                np.concatenate(data_out),
                np.concatenate(indices_out),
                np.asarray(indptr),
            ),
            shape=similarity.shape,
        ),
        items,
    )


def build_profile(
    positive: pd.DataFrame,
    target_ids: set[int],
    catalog_items: np.ndarray,
    history_end: int,
) -> dict[int, list[tuple[int, float]]]:
    """Build play-, provenance-, and recency-weighted user profiles."""
    history = (
        positive[positive.uid.isin(target_ids) & positive.item_id.isin(catalog_items)]
        .groupby(["uid", "item_id"], sort=False)
        .agg(
            plays=("item_id", "size"),
            last=("timestamp", "max"),
            organic=("is_organic", "mean"),
        )
        .reset_index()
    )
    history["weight"] = (
        (1.0 + np.log1p(history.plays))
        * (1.0 + history.organic)
        * np.exp(-(history_end - history["last"]) / (120 * DAY))
    )
    ordered = history.sort_values(["uid", "weight"], ascending=[True, False])
    return {
        int(uid): list(zip(group.item_id.astype(int), group.weight.astype(float)))[
            :PROFILE_DEPTH
        ]
        for uid, group in ordered.groupby("uid", sort=False)
    }


def load_history(data: Path, stage: str) -> tuple[pd.DataFrame, int, str]:
    columns = ["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"]
    train = pd.read_parquet(data / "train_listens.parquet", columns=columns)
    if stage == "validation":
        return train, TRAIN_END, "validation_preference_state.parquet"
    validation = pd.read_parquet(data / "val_listens.parquet", columns=columns)
    return (
        pd.concat([train, validation], ignore_index=True),
        VALIDATION_END,
        "test_preference_state.parquet",
    )


def build_reference(public: Path, stage: str) -> pd.DataFrame:
    data = public / "data"
    history, history_end, state_name = load_history(data, stage)
    positive = history[history.played_ratio_pct >= LISTEN_POSITIVE]
    catalog = pd.read_parquet(data / "candidate_catalog.parquet").sort_values(
        "popularity_rank"
    )
    candidate_items = catalog.item_id.to_numpy(dtype=np.int64, copy=False)
    targets = pd.read_parquet(data / "target_users.parquet", columns=["uid"])
    target_ids = set(targets.uid.astype(int))
    state = pd.read_parquet(data / state_name)
    active_likes = state[state.state == "liked"][["uid", "item_id"]].copy()

    print(f"fitting {stage} reference similarity", file=sys.stderr, flush=True)
    similarity, item_ids = fit_similarity(
        positive, active_likes, candidate_items, history_end
    )
    profile = build_profile(positive, target_ids, candidate_items, history_end)
    candidate_name = (
        "validation_eligible_candidates.npz"
        if stage == "validation"
        else "test_eligible_candidates.npz"
    )
    eligibility = CandidateSet(data / candidate_name)
    index_of = {int(value): index for index, value in enumerate(item_ids)}

    rows: list[tuple[int, int, int]] = []
    for position, raw_uid in enumerate(eligibility.uids):
        uid = int(raw_uid)
        scores: dict[int, float] = {}
        for item, weight in profile.get(uid, []):
            column = index_of.get(item)
            if column is None:
                continue
            start, end = similarity.indptr[column], similarity.indptr[column + 1]
            for neighbour, value in zip(
                similarity.indices[start:end], similarity.data[start:end]
            ):
                candidate = int(item_ids[neighbour])
                scores[candidate] = scores.get(candidate, 0.0) + weight * float(value)
        ordered = [
            item for item, _ in sorted(scores.items(), key=lambda pair: -pair[1])
        ]
        ranked = eligibility.filter_ranked(uid, ordered).tolist()
        selected = set(ranked)
        if len(ranked) < TOP_N:
            for raw_item in eligibility.candidates_for(uid):
                item = int(raw_item)
                if item not in selected:
                    ranked.append(item)
                    selected.add(item)
                if len(ranked) >= TOP_N:
                    break
        ranked = ranked[:TOP_N]
        if len(ranked) != TOP_N or len(set(ranked)) != TOP_N:
            raise RuntimeError(f"reference ranking invalid for uid {uid}")
        rows.extend((uid, item, rank) for rank, item in enumerate(ranked, start=1))
        if position and position % 1_000 == 0:
            print(
                f"ranked {position:,}/{len(eligibility.uids):,} users",
                file=sys.stderr,
                flush=True,
            )
    return pd.DataFrame(rows, columns=["uid", "item_id", "rank"])


def write_ranking(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        frame.to_parquet(output, index=False, compression="zstd")
    elif output.suffix == ".csv":
        frame.to_csv(output, index=False)
    else:
        raise ValueError("--output must end in .parquet or .csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("validation", "submission"), default="submission"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=Path("/task"),
        help="solver-visible public package directory",
    )
    args = parser.parse_args()

    public = args.public_dir.resolve()
    output = args.output.resolve()

    ranking = build_reference(public, args.stage)
    write_ranking(ranking, output)
    result: dict[str, object] = {
        "stage": args.stage,
        "output": str(output),
        "rows": len(ranking),
        "users": int(ranking.uid.nunique()),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
