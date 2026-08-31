#!/usr/bin/env python3
"""Build cutoff-specific candidate eligibility from public event history.

An item is eligible for a target user when it belongs to the candidate catalog
and that user has not encountered it in any event family visible at the ranking
cutoff. All event families count as encounters, regardless of whether the event
was positive, negative, organic, or recommendation-driven.

The script expects a sibling ``data`` directory containing ``target_users`` and
``candidate_catalog`` parquet files plus the ``train_*`` and ``val_*`` event
tables. It writes two compressed NumPy archives beside this file:

* ``validation_eligible_candidates.npz`` excludes ``train_*`` encounters.
* ``submission_eligible_candidates.npz`` excludes both ``train_*`` and
  ``val_*`` encounters. ``build_task.py`` later publishes this artifact as
  ``test_eligible_candidates.npz``.

Each archive contains sorted ``uids``, popularity-ordered ``item_ids``, and a
two-dimensional ``uint8`` array named ``packed_eligible``. Item column ``c`` is
stored in byte ``c // 8`` at bit ``c % 8``; one means eligible and zero means
excluded. Equivalently, recover a user's mask with::

    np.unpackbits(
        packed_eligible[user_row], count=len(item_ids), bitorder="little"
    )

Run this module without arguments from any working directory. Input and output
paths are resolved relative to the module itself.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
EVENT_NAMES = ("listens", "likes", "dislikes", "unlikes", "undislikes")
BATCH_SIZE = 1_000_000


def empty_mask(n_users: int, n_items: int) -> np.ndarray:
    """Return a packed bitmap in which every real item starts eligible.

    The returned array has shape ``(n_users, ceil(n_items / 8))`` and dtype
    ``uint8``. Bits use the same little-endian convention as
    ``numpy.unpackbits(..., bitorder="little")``. Unused bits in the final byte
    are cleared so they cannot be mistaken for catalog items when rows are
    counted without an explicit item limit.

    Args:
        n_users: Number of target-user rows in the bitmap.
        n_items: Number of candidate-catalog columns represented by each row.

    Returns:
        The initialized packed eligibility bitmap.
    """
    width = (n_items + 7) // 8
    packed = np.full((n_users, width), 255, dtype=np.uint8)
    if n_items % 8:
        packed[:, -1] &= np.uint8((1 << (n_items % 8)) - 1)
    return packed


def exclude_file(
    packed: np.ndarray,
    path: Path,
    uid_index: pd.Index,
    item_index: pd.Index,
) -> None:
    """Clear bits for target-user/catalog-item pairs found in one event file.

    The parquet file is streamed in batches and only its ``uid`` and
    ``item_id`` columns are read. Rows for non-target users or items outside the
    candidate catalog are ignored. Duplicate encounters are harmless.

    Args:
        packed: Eligibility bitmap to mutate in place. Its rows and columns
            must align with ``uid_index`` and ``item_index`` respectively.
        path: Event parquet file containing ``uid`` and ``item_id`` columns.
        uid_index: Target user IDs in bitmap row order.
        item_index: Candidate item IDs in bitmap column order.
    """
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE, columns=["uid", "item_id"]
    ):
        raw_uids = batch.column(0).to_numpy(zero_copy_only=False)
        raw_items = batch.column(1).to_numpy(zero_copy_only=False)
        rows = uid_index.get_indexer(raw_uids)
        columns = item_index.get_indexer(raw_items)
        matched = (rows >= 0) & (columns >= 0)
        if not np.any(matched):
            continue
        rows = rows[matched]
        columns = columns[matched]
        byte_columns = columns // 8
        clear_masks = np.invert(
            (1 << (columns % 8)).astype(np.uint8, copy=False)
        )
        np.bitwise_and.at(packed, (rows, byte_columns), clear_masks)


def build() -> None:
    """Create eligibility archives for the validation and submission cutoffs.

    Target users are sorted by ID, while candidate items retain ascending
    ``popularity_rank`` order. For each cutoff, the function starts with every
    catalog item eligible and clears pairs encountered in any applicable event
    table. It refuses to write an artifact if any target user would have fewer
    than 100 candidates, the number required for a submission.

    The resulting archives contain ``uids`` and ``item_ids`` as ``uint32``
    arrays plus the ``packed_eligible`` bitmap documented at module level.

    Raises:
        RuntimeError: If a target user has fewer than 100 eligible candidates
            at either cutoff.
    """
    targets = pd.read_parquet(DATA / "target_users.parquet", columns=["uid"])
    catalog = pd.read_parquet(
        DATA / "candidate_catalog.parquet",
        columns=["item_id", "popularity_rank"],
    ).sort_values("popularity_rank")
    uids = np.sort(targets.uid.to_numpy(dtype=np.int64, copy=False))
    item_ids = catalog.item_id.to_numpy(dtype=np.int64, copy=False)
    uid_index = pd.Index(uids)
    item_index = pd.Index(item_ids)

    stages = {
        "validation": ("train",),
        "submission": ("train", "val"),
    }
    for stage, prefixes in stages.items():
        packed = empty_mask(len(uids), len(item_ids))
        for prefix in prefixes:
            for name in EVENT_NAMES:
                exclude_file(
                    packed,
                    DATA / f"{prefix}_{name}.parquet",
                    uid_index,
                    item_index,
                )
        bit_counts = np.unpackbits(packed, axis=1, bitorder="little").sum(axis=1)
        if int(bit_counts.min()) < 100:
            raise RuntimeError(f"a target user has fewer than 100 {stage} candidates")
        np.savez_compressed(
            HERE / f"{stage}_eligible_candidates.npz",
            uids=uids.astype(np.uint32, copy=False),
            item_ids=item_ids.astype(np.uint32, copy=False),
            packed_eligible=packed,
        )
        print(
            f"{stage} eligibility users={len(uids):,} items={len(item_ids):,} "
            f"excluded_pairs={int(len(uids) * len(item_ids) - bit_counts.sum()):,}"
        )


if __name__ == "__main__":
    build()
