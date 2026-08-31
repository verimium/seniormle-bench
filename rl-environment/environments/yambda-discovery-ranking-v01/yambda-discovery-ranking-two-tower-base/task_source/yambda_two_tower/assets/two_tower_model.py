"""Shared public-data primitives for the Yambda two-tower baseline."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn


DAY = 86_400
TRAIN_END = 24_786_800
VALIDATION_END = 25_393_400
LISTEN_POSITIVE_THRESHOLD = 50
PROFILE_LOOKBACK_DAYS = 120
LIKE_ITEMS = 48
DISLIKE_ITEMS = 24
TOP_N = 100


@dataclass(frozen=True)
class TowerConfig:
    sequence_length: int = 100
    embedding_dim: int = 192
    hidden_dim: int = 192
    user_hidden_1: int = 512
    user_hidden_2: int = 256
    item_hidden_1: int = 512
    item_hidden_2: int = 256
    dropout: float = 0.05
    epochs: int = 1
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    gradient_clip: float = 2.0
    uniform_negatives: int = 24
    inbatch_negatives: int = 8
    inbatch_correction_power: float = 0.5
    inbatch_correction_min: float = 0.25
    inbatch_correction_max: float = 4.0
    seed: int = 29


@dataclass(frozen=True)
class ListenEvents:
    timestamps: np.ndarray
    items: np.ndarray
    organic: np.ndarray
    completion_pct: np.ndarray


@dataclass(frozen=True)
class ExplicitEvents:
    timestamps: np.ndarray
    items: np.ndarray
    channel: np.ndarray
    active: np.ndarray
    organic_like: np.ndarray


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def build_item_features(catalog: pd.DataFrame) -> np.ndarray:
    """Return four standardized features built only from the public catalog."""
    positives = catalog.positive_listens.to_numpy(dtype=np.float64, copy=False)
    organic = catalog.organic_positive_listens.to_numpy(dtype=np.float64, copy=False)
    popularity_rank = catalog.popularity_rank.to_numpy(dtype=np.float64, copy=False)
    features = np.column_stack(
        [
            np.log1p(positives),
            np.log1p(organic),
            organic / np.maximum(positives, 1.0),
            1.0 - (popularity_rank - 1.0) / max(len(catalog) - 1, 1),
        ]
    )
    means = features.mean(axis=0, keepdims=True)
    scales = np.maximum(features.std(axis=0, keepdims=True), 1.0e-6)
    standardized = ((features - means) / scales).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise RuntimeError("catalog features contain non-finite values")
    return standardized


def _combine_listen_chunks(parts: list[ListenEvents]) -> ListenEvents:
    if len(parts) == 1:
        return parts[0]
    timestamps = np.concatenate([part.timestamps for part in parts])
    items = np.concatenate([part.items for part in parts])
    organic = np.concatenate([part.organic for part in parts])
    completion = np.concatenate([part.completion_pct for part in parts])
    order = np.argsort(timestamps, kind="stable")
    return ListenEvents(
        timestamps=timestamps[order],
        items=items[order],
        organic=organic[order],
        completion_pct=completion[order],
    )


def iter_listen_users(
    path: Path, item_index: pd.Index
) -> Iterator[tuple[int, ListenEvents]]:
    """Stream one uid-sorted public listen file one user at a time."""
    parquet = pq.ParquetFile(path)
    pending_uid: int | None = None
    pending: list[ListenEvents] = []
    previous_uid = -1
    for batch in parquet.iter_batches(
        batch_size=750_000,
        columns=["uid", "timestamp", "item_id", "is_organic", "played_ratio_pct"],
    ):
        frame = batch.to_pandas()
        mapped = item_index.get_indexer(frame.item_id.to_numpy(dtype=np.int64))
        keep = mapped >= 0
        if not np.any(keep):
            continue
        uids = frame.uid.to_numpy(dtype=np.int64, copy=False)[keep]
        timestamps = frame.timestamp.to_numpy(dtype=np.int64, copy=False)[keep]
        items = mapped[keep].astype(np.int32, copy=False)
        organic = frame.is_organic.to_numpy(dtype=np.uint8, copy=False)[keep]
        completion = frame.played_ratio_pct.to_numpy(dtype=np.uint16, copy=False)[keep]
        if np.any(uids[1:] < uids[:-1]):
            raise RuntimeError(f"{path.name} is not sorted by uid")
        boundaries = np.r_[
            0,
            np.flatnonzero(uids[1:] != uids[:-1]) + 1,
            len(uids),
        ]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            uid = int(uids[start])
            part = ListenEvents(
                timestamps[start:end].copy(),
                items[start:end].copy(),
                organic[start:end].copy(),
                completion[start:end].copy(),
            )
            if pending_uid is None:
                pending_uid = uid
            elif uid != pending_uid:
                if pending_uid < previous_uid:
                    raise RuntimeError(f"{path.name} user order regressed")
                yield pending_uid, _combine_listen_chunks(pending)
                previous_uid = pending_uid
                pending_uid = uid
                pending = []
            pending.append(part)
    if pending_uid is not None:
        yield pending_uid, _combine_listen_chunks(pending)


def merge_listen_streams(
    paths: list[Path], item_index: pd.Index
) -> Iterator[tuple[int, ListenEvents]]:
    """Merge uid-sorted public listen files without materializing them."""
    streams = [iter(iter_listen_users(path, item_index)) for path in paths]
    heads: list[tuple[int, ListenEvents] | None] = [
        next(stream, None) for stream in streams
    ]
    while any(head is not None for head in heads):
        uid = min(head[0] for head in heads if head is not None)
        parts: list[ListenEvents] = []
        for position, head in enumerate(heads):
            if head is not None and head[0] == uid:
                parts.append(head[1])
                heads[position] = next(streams[position], None)
        yield uid, _combine_listen_chunks(parts)


def load_explicit_events(
    data: Path,
    prefixes: tuple[str, ...],
    item_index: pd.Index,
) -> dict[int, ExplicitEvents]:
    """Load public explicit actions as chronological state changes."""
    specifications = (
        ("likes", 0, 1, 0),
        ("unlikes", 0, 0, 1),
        ("dislikes", 1, 1, 0),
        ("undislikes", 1, 0, 1),
    )
    parts: list[pd.DataFrame] = []
    for family, channel, active, tie_priority in specifications:
        for prefix in prefixes:
            frame = pd.read_parquet(
                data / f"{prefix}_{family}.parquet",
                columns=["uid", "timestamp", "item_id", "is_organic"],
            )
            mapped = item_index.get_indexer(
                frame.item_id.to_numpy(dtype=np.int64, copy=False)
            )
            keep = mapped >= 0
            if not np.any(keep):
                continue
            retained = frame.loc[keep, ["uid", "timestamp", "is_organic"]].copy()
            retained["item"] = mapped[keep].astype(np.int32, copy=False)
            retained["channel"] = np.int8(channel)
            retained["active"] = np.int8(active)
            retained["tie_priority"] = np.int8(tie_priority)
            retained["organic_like"] = np.int8(
                family == "likes"
            ) * retained.is_organic.astype(np.int8)
            parts.append(retained.drop(columns="is_organic"))
    if not parts:
        return {}
    events = pd.concat(parts, ignore_index=True)
    events = events.sort_values(
        ["uid", "timestamp", "tie_priority"], kind="mergesort"
    ).reset_index(drop=True)
    result: dict[int, ExplicitEvents] = {}
    for uid, rows in events.groupby("uid", sort=False):
        result[int(uid)] = ExplicitEvents(
            timestamps=rows.timestamp.to_numpy(dtype=np.int64, copy=True),
            items=rows.item.to_numpy(dtype=np.int32, copy=True),
            channel=rows.channel.to_numpy(dtype=np.int8, copy=True),
            active=rows.active.to_numpy(dtype=np.int8, copy=True),
            organic_like=rows.organic_like.to_numpy(dtype=np.int8, copy=True),
        )
    log(f"loaded explicit events: rows={len(events):,} users={len(result):,}")
    return result


def empty_explicit_events() -> ExplicitEvents:
    return ExplicitEvents(
        timestamps=np.empty(0, dtype=np.int64),
        items=np.empty(0, dtype=np.int32),
        channel=np.empty(0, dtype=np.int8),
        active=np.empty(0, dtype=np.int8),
        organic_like=np.empty(0, dtype=np.int8),
    )


def pack_active_items(
    active: dict[int, int], width: int, padding_item: int
) -> np.ndarray:
    packed = np.full(width, padding_item, dtype=np.uint16)
    if not active:
        return packed
    ordered = sorted(active.items(), key=lambda pair: (-pair[1], pair[0]))[:width]
    packed[: len(ordered)] = np.fromiter(
        (item for item, _ in ordered), dtype=np.uint16, count=len(ordered)
    )
    return packed


def load_inference_state(
    path: Path,
    item_index: pd.Index,
    uids: np.ndarray,
    n_items: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack the task-provided active explicit state."""
    state = pd.read_parquet(path, columns=["uid", "item_id", "state"])
    mapped = item_index.get_indexer(state.item_id.to_numpy(dtype=np.int64))
    keep = mapped >= 0
    state = state.loc[keep, ["uid", "state"]].copy()
    state["item"] = mapped[keep].astype(np.int32, copy=False)
    user_rows = pd.Index(uids)
    liked = np.full((len(uids), LIKE_ITEMS), n_items, dtype=np.uint16)
    disliked = np.full((len(uids), DISLIKE_ITEMS), n_items, dtype=np.uint16)
    like_counts = np.zeros(len(uids), dtype=np.int32)
    dislike_counts = np.zeros(len(uids), dtype=np.int32)
    for value, width, destination, counts in (
        ("liked", LIKE_ITEMS, liked, like_counts),
        ("disliked", DISLIKE_ITEMS, disliked, dislike_counts),
    ):
        selected = state[state.state == value]
        for uid, rows in selected.groupby("uid", sort=False):
            row = int(user_rows.get_indexer([int(uid)])[0])
            if row < 0:
                continue
            items = np.sort(rows.item.unique())
            counts[row] = len(items)
            retained = items[:width]
            destination[row, : len(retained)] = retained.astype(np.uint16)
    return liked, disliked, like_counts, dislike_counts


class SequentialTwoTower(nn.Module):
    """Order-sensitive user tower and independent catalog-aware item tower."""

    def __init__(self, item_features: np.ndarray, config: TowerConfig) -> None:
        super().__init__()
        n_items, n_item_features = item_features.shape
        dim = config.embedding_dim
        self.n_items = n_items
        self.context_embedding = nn.Embedding(
            n_items + 1, dim, padding_idx=n_items
        )
        self.event_projection = nn.Sequential(
            nn.Linear(dim + 3, config.hidden_dim),
            nn.GELU(),
        )
        self.sequence_encoder = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
        )
        self.user_tower = nn.Sequential(
            nn.Linear(dim * 3 + 6, config.user_hidden_1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.user_hidden_1, config.user_hidden_2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.user_hidden_2, dim),
        )
        self.candidate_embedding = nn.Embedding(n_items, dim)
        self.register_buffer(
            "item_features", torch.from_numpy(item_features), persistent=True
        )
        self.item_tower = nn.Sequential(
            nn.Linear(dim + n_item_features, config.item_hidden_1),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.item_hidden_1, config.item_hidden_2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.item_hidden_2, dim),
        )
        nn.init.normal_(self.context_embedding.weight, std=0.02)
        nn.init.normal_(self.candidate_embedding.weight, std=0.02)
        with torch.no_grad():
            self.context_embedding.weight[n_items].zero_()

    def _mean_context_channel(self, items: torch.Tensor) -> torch.Tensor:
        mask = items.ne(self.n_items)
        vectors = self.context_embedding(items)
        return (vectors * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)

    def encode_items(self, items: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat(
            [self.candidate_embedding(items), self.item_features[items]], dim=1
        )
        return self.item_tower(inputs)


def invalid_negative_mask(
    negatives: torch.Tensor,
    targets: torch.Tensor,
    context_items: torch.Tensor,
    liked_items: torch.Tensor,
) -> torch.Tensor:
    """Identify targets, known positives, active likes, and row duplicates."""
    invalid = negatives.eq(targets[:, None])
    invalid |= negatives[:, :, None].eq(context_items[:, None, :]).any(dim=2)
    invalid |= negatives[:, :, None].eq(liked_items[:, None, :]).any(dim=2)
    for column in range(1, negatives.shape[1]):
        invalid[:, column] |= negatives[:, column, None].eq(
            negatives[:, :column]
        ).any(dim=1)
    return invalid


def draw_filtered_mixed_negatives(
    targets: torch.Tensor,
    context_items: torch.Tensor,
    liked_items: torch.Tensor,
    n_items: int,
    uniform_count: int,
    inbatch_count: int,
    item_correction: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw uniform and unique-target in-batch negatives."""
    batch_size = len(targets)
    uniform = torch.randint(
        n_items,
        (batch_size, uniform_count),
        generator=generator,
        device="cpu",
    )
    inbatch = torch.empty((batch_size, 0), dtype=torch.long, device="cpu")
    corrected_count = 0
    if inbatch_count:
        unique_targets = torch.unique(targets.detach().cpu())
        corrected_count = min(inbatch_count, max(len(unique_targets) - 1, 0))
        if corrected_count:
            priorities = torch.rand(
                (batch_size, len(unique_targets)), generator=generator
            )
            priorities.masked_fill_(
                unique_targets[None, :].eq(targets.detach().cpu()[:, None]),
                -1.0,
            )
            columns = torch.topk(priorities, k=corrected_count, dim=1).indices
            inbatch = unique_targets[columns]
        if corrected_count < inbatch_count:
            filler = torch.randint(
                n_items,
                (batch_size, inbatch_count - corrected_count),
                generator=generator,
                device="cpu",
            )
            inbatch = torch.cat([inbatch, filler], dim=1)

    negatives_cpu = torch.cat([uniform, inbatch], dim=1)
    correction_cpu = torch.ones_like(negatives_cpu, dtype=torch.float32)
    if corrected_count:
        correction_cpu[
            :, uniform_count : uniform_count + corrected_count
        ] = item_correction[inbatch[:, :corrected_count]]
    negatives = negatives_cpu.to(device)
    correction = correction_cpu.to(device)
    for _ in range(32):
        invalid = invalid_negative_mask(
            negatives, targets, context_items, liked_items
        )
        if not bool(invalid.any()):
            return negatives, correction
        replacements = torch.randint(
            n_items,
            (int(invalid.sum().detach().cpu()),),
            generator=generator,
            device="cpu",
        ).to(device)
        negatives[invalid] = replacements
        correction[invalid] = 1.0
    invalid = invalid_negative_mask(
        negatives, targets, context_items, liked_items
    )
    if bool(invalid.any()):
        raise RuntimeError("mixed negative resampling failed false-negative filter")
    return negatives, correction
