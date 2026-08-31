#!/usr/bin/env python3
"""Train the Yambda windowed two-tower baseline.

Every catalog listen with at least 50 percent completion is a sequence token.
Within each selected chronological window, every adjacent target contributes a
loss.  The fast default retains each user's most recent 128 training targets and
16 held-out validation targets; passing zero for either cap restores the
uncapped behavior for that side of the split.  A 100-step truncated-BPTT block
computes all of its timestep losses and carries the detached GRU hidden state
into the next block.

The model is a pure two-tower retriever: no user-ID embedding, no output
normalization, raw dot-product retrieval, sample-weighted binary cross entropy,
and 24 uniform plus eight propensity-corrected unique in-batch negatives. Every
selected adjacent target retains a nonzero weight. Only files under the supplied
public directory are read.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import two_tower_model as base


CACHE_VERSION = 2


@dataclass(frozen=True)
class Config:
    bptt_steps: int = 100
    user_batch_size: int = 64
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
    loss_microbatch_size: int = 65_536
    uniform_negatives: int = 24
    inbatch_negatives: int = 8
    inbatch_correction_power: float = 0.5
    inbatch_correction_min: float = 0.25
    inbatch_correction_max: float = 4.0
    negative_filter_scope: str = "full-training-history"
    target_weighting: str = "organic-strong"
    ordinary_target_weight: float = 0.25
    organic_strong_target_weight: float = 1.0
    organic_like_target_weight: float = 3.0
    discovery_horizon_days: int = 7
    validation_fraction: float = 0.10
    max_train_targets_per_user: int = 128
    max_validation_targets_per_user: int = 16
    early_stopping_patience: int = 1
    early_stopping_min_delta: float = 1.0e-4
    seed: int = 29


@dataclass(frozen=True)
class CandidateSet:
    uids: np.ndarray
    item_ids: np.ndarray
    packed: np.ndarray

    @classmethod
    def read(cls, path: Path) -> "CandidateSet":
        with np.load(path, allow_pickle=False) as artifact:
            result = cls(
                uids=artifact["uids"].astype(np.int64, copy=True),
                item_ids=artifact["item_ids"].astype(np.int64, copy=True),
                packed=artifact["packed_eligible"].astype(np.uint8, copy=True),
            )
        expected = (len(result.uids), (len(result.item_ids) + 7) // 8)
        if result.packed.shape != expected:
            raise ValueError(
                f"invalid candidate artifact shape {result.packed.shape}; "
                f"expected {expected}"
            )
        return result


@dataclass(frozen=True)
class SequenceData:
    item_ids: np.ndarray
    item_features: np.ndarray
    user_ids: np.ndarray
    offsets: np.ndarray
    sequence_timestamps: np.ndarray
    sequence_items: np.ndarray
    event_features: np.ndarray
    activity_features: np.ndarray
    state_ids: np.ndarray
    liked_states: np.ndarray
    disliked_states: np.ndarray
    state_counts: np.ndarray
    inference_user_rows: np.ndarray
    inference_liked_items: np.ndarray
    inference_disliked_items: np.ndarray
    inference_user_features: np.ndarray

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def target_count(self) -> int:
        return int(np.maximum(self.lengths - 1, 0).sum())


@dataclass(frozen=True)
class TargetPartition:
    """A contiguous recent target window and its chronological BCE split."""

    window_starts: np.ndarray
    training_counts: np.ndarray
    validation_counts: np.ndarray
    history_training_counts: np.ndarray

    @property
    def selected_count(self) -> int:
        return int(self.training_counts.sum() + self.validation_counts.sum())


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_config(config: Config) -> base.TowerConfig:
    """Map the dense trainer configuration onto the unchanged tower."""
    return base.TowerConfig(
        sequence_length=config.bptt_steps,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        user_hidden_1=config.user_hidden_1,
        user_hidden_2=config.user_hidden_2,
        item_hidden_1=config.item_hidden_1,
        item_hidden_2=config.item_hidden_2,
        dropout=config.dropout,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
        uniform_negatives=config.uniform_negatives,
        inbatch_negatives=config.inbatch_negatives,
        inbatch_correction_power=config.inbatch_correction_power,
        inbatch_correction_min=config.inbatch_correction_min,
        inbatch_correction_max=config.inbatch_correction_max,
        seed=config.seed,
    )


def append_state_snapshot(
    active_likes: dict[int, int],
    active_dislikes: dict[int, int],
    n_items: int,
    liked_states: list[np.ndarray],
    disliked_states: list[np.ndarray],
    state_counts: list[tuple[int, int]],
) -> int:
    liked_states.append(base.pack_active_items(active_likes, base.LIKE_ITEMS, n_items))
    disliked_states.append(
        base.pack_active_items(active_dislikes, base.DISLIKE_ITEMS, n_items)
    )
    state_counts.append((len(active_likes), len(active_dislikes)))
    return len(liked_states) - 1


def causal_activity_features(
    timestamps: np.ndarray,
    items: np.ndarray,
    organic: np.ndarray,
    completion: np.ndarray,
) -> np.ndarray:
    """Compute the first four reviewed user features at every event cutoff."""
    length = len(items)
    result = np.zeros((length, 4), dtype=np.float16)
    if not length:
        return result
    left_rows = np.searchsorted(
        timestamps,
        timestamps - base.PROFILE_LOOKBACK_DAYS * base.DAY,
        side="left",
    )
    rows = np.arange(length, dtype=np.int64)
    counts = rows - left_rows + 1
    organic_sum = np.cumsum(organic.astype(np.float64))
    completion_sum = np.cumsum(np.minimum(completion, 100).astype(np.float64))
    before_organic = np.where(
        left_rows > 0, organic_sum[np.maximum(left_rows - 1, 0)], 0.0
    )
    before_completion = np.where(
        left_rows > 0, completion_sum[np.maximum(left_rows - 1, 0)], 0.0
    )

    unique_counts = np.empty(length, dtype=np.int32)
    frequencies: dict[int, int] = {}
    left = 0
    for right, raw_item in enumerate(items):
        item = int(raw_item)
        frequencies[item] = frequencies.get(item, 0) + 1
        required_left = int(left_rows[right])
        while left < required_left:
            old_item = int(items[left])
            remaining = frequencies[old_item] - 1
            if remaining:
                frequencies[old_item] = remaining
            else:
                del frequencies[old_item]
            left += 1
        unique_counts[right] = len(frequencies)

    result[:, 0] = (np.log1p(counts) / 8.0).astype(np.float16)
    result[:, 1] = (np.log1p(unique_counts) / 6.0).astype(np.float16)
    result[:, 2] = (
        (organic_sum - before_organic) / np.maximum(counts, 1)
    ).astype(np.float16)
    result[:, 3] = (
        (completion_sum - before_completion)
        / np.maximum(counts, 1)
        / 100.0
    ).astype(np.float16)
    return result


def build_sequence_parts(
    listens: base.ListenEvents,
    explicit: base.ExplicitEvents,
    n_items: int,
    liked_states: list[np.ndarray],
    disliked_states: list[np.ndarray],
    state_counts: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    qualifying = listens.completion_pct >= base.LISTEN_POSITIVE_THRESHOLD
    timestamps = listens.timestamps[qualifying].astype(np.int64, copy=False)
    items = listens.items[qualifying].astype(np.uint16, copy=False)
    organic = listens.organic[qualifying].astype(np.float32, copy=False)
    completion = listens.completion_pct[qualifying].astype(np.float32, copy=False)
    if not len(items):
        return None

    event_features = np.zeros((len(items), 3), dtype=np.float16)
    event_features[:, 0] = organic.astype(np.float16)
    event_features[:, 1] = (
        np.minimum(completion, 100.0) / 100.0
    ).astype(np.float16)
    if len(items) > 1:
        gaps = np.maximum(0, timestamps[1:] - timestamps[:-1])
        event_features[1:, 2] = np.exp(
            -gaps.astype(np.float64) / (60.0 * base.DAY)
        ).astype(np.float16)

    activity = causal_activity_features(
        timestamps, items, organic, completion
    )
    states = np.zeros(len(items), dtype=np.uint32)
    active_likes: dict[int, int] = {}
    active_dislikes: dict[int, int] = {}
    explicit_position = 0
    current_state = 0
    for row, timestamp in enumerate(timestamps):
        changed = False
        while (
            explicit_position < len(explicit.timestamps)
            and int(explicit.timestamps[explicit_position]) < int(timestamp)
        ):
            item = int(explicit.items[explicit_position])
            event_timestamp = int(explicit.timestamps[explicit_position])
            destination = (
                active_likes
                if int(explicit.channel[explicit_position]) == 0
                else active_dislikes
            )
            if int(explicit.active[explicit_position]) == 1:
                destination[item] = event_timestamp
            else:
                destination.pop(item, None)
            explicit_position += 1
            changed = True
        if changed:
            current_state = append_state_snapshot(
                active_likes,
                active_dislikes,
                n_items,
                liked_states,
                disliked_states,
                state_counts,
            )
        states[row] = np.uint32(current_state)
    return timestamps, items.copy(), event_features, activity, states


def final_user_features(
    timestamps: np.ndarray,
    items: np.ndarray,
    event_features: np.ndarray,
    history_end: int,
    like_count: int,
    dislike_count: int,
) -> np.ndarray:
    start = int(
        np.searchsorted(
            timestamps,
            history_end - base.PROFILE_LOOKBACK_DAYS * base.DAY,
            side="left",
        )
    )
    result = np.zeros(6, dtype=np.float32)
    recent_items = items[start:]
    if len(recent_items):
        result[0] = np.log1p(len(recent_items)) / 8.0
        result[1] = np.log1p(len(np.unique(recent_items))) / 6.0
        result[2] = float(np.mean(event_features[start:, 0]))
        result[3] = float(np.mean(event_features[start:, 1]))
    result[4] = np.log1p(like_count) / 6.0
    result[5] = np.log1p(dislike_count) / 4.0
    return result


def prepare_data(
    public: Path, stage: str, config: Config
) -> tuple[SequenceData, CandidateSet]:
    data = public / "data"
    catalog = pd.read_parquet(data / "candidate_catalog.parquet").sort_values(
        "popularity_rank"
    )
    item_ids = catalog.item_id.to_numpy(dtype=np.int64, copy=True)
    item_index = pd.Index(item_ids)
    n_items = len(item_ids)
    if n_items >= np.iinfo(np.uint16).max:
        raise RuntimeError("uint16 item storage is too small for this catalog")

    if stage == "validation":
        prefixes = ("train",)
        history_end = base.TRAIN_END
        candidate_name = "validation_eligible_candidates.npz"
        state_name = "validation_preference_state.parquet"
    else:
        prefixes = ("train", "val")
        history_end = base.VALIDATION_END
        candidate_name = "test_eligible_candidates.npz"
        state_name = "test_preference_state.parquet"
    eligibility = CandidateSet.read(data / candidate_name)
    if not np.array_equal(eligibility.item_ids, item_ids):
        raise RuntimeError("catalog and candidate artifact item order differ")

    explicit_by_user = base.load_explicit_events(data, prefixes, item_index)
    inference_likes, inference_dislikes, like_counts, dislike_counts = (
        base.load_inference_state(
            data / state_name,
            item_index,
            eligibility.uids,
            n_items,
        )
    )
    inference_rows = {int(uid): row for row, uid in enumerate(eligibility.uids)}
    inference_user_features = np.zeros((len(eligibility.uids), 6), dtype=np.float32)
    inference_sequence_rows = np.full(len(eligibility.uids), -1, dtype=np.int32)

    # Snapshot zero is the shared empty explicit state.
    liked_states = [np.full(base.LIKE_ITEMS, n_items, dtype=np.uint16)]
    disliked_states = [
        np.full(base.DISLIKE_ITEMS, n_items, dtype=np.uint16)
    ]
    state_counts: list[tuple[int, int]] = [(0, 0)]
    user_ids: list[int] = []
    offsets = [0]
    timestamp_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []
    event_parts: list[np.ndarray] = []
    activity_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []

    listen_paths = [data / f"{prefix}_listens.parquet" for prefix in prefixes]
    started = time.monotonic()
    for seen_users, (uid, listens) in enumerate(
        base.merge_listen_streams(listen_paths, item_index), start=1
    ):
        parts = build_sequence_parts(
            listens,
            explicit_by_user.get(uid, base.empty_explicit_events()),
            n_items,
            liked_states,
            disliked_states,
            state_counts,
        )
        if parts is None:
            continue
        sequence_row = len(user_ids)
        user_ids.append(uid)
        timestamp_parts.append(parts[0].astype(np.uint32, copy=False))
        item_parts.append(parts[1])
        event_parts.append(parts[2])
        activity_parts.append(parts[3])
        state_parts.append(parts[4])
        offsets.append(offsets[-1] + len(parts[1]))
        inference_row = inference_rows.get(uid)
        if inference_row is not None:
            inference_sequence_rows[inference_row] = sequence_row
            inference_user_features[inference_row] = final_user_features(
                parts[0],
                parts[1],
                parts[2],
                history_end,
                int(like_counts[inference_row]),
                int(dislike_counts[inference_row]),
            )
        if seen_users % 1_000 == 0:
            log(
                f"prepared users={seen_users:,} tokens={offsets[-1]:,} "
                f"states={len(liked_states):,} "
                f"elapsed={time.monotonic() - started:.1f}s"
            )

    missing = np.flatnonzero(inference_sequence_rows < 0)
    if len(missing):
        inference_user_features[missing, 4] = np.log1p(like_counts[missing]) / 6.0
        inference_user_features[missing, 5] = (
            np.log1p(dislike_counts[missing]) / 4.0
        )
        log(f"target users without qualifying catalog sequence={len(missing):,}")

    prepared = SequenceData(
        item_ids=item_ids,
        item_features=base.build_item_features(catalog),
        user_ids=np.asarray(user_ids, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        sequence_timestamps=np.concatenate(timestamp_parts),
        sequence_items=np.concatenate(item_parts),
        event_features=np.concatenate(event_parts),
        activity_features=np.concatenate(activity_parts),
        state_ids=np.concatenate(state_parts),
        liked_states=np.stack(liked_states),
        disliked_states=np.stack(disliked_states),
        state_counts=np.asarray(state_counts, dtype=np.int32),
        inference_user_rows=inference_sequence_rows,
        inference_liked_items=inference_likes,
        inference_disliked_items=inference_dislikes,
        inference_user_features=inference_user_features,
    )
    log(
        f"prepared stage={stage} sequence_users={len(prepared.user_ids):,} "
        f"tokens={len(prepared.sequence_items):,} "
        f"next_item_targets={prepared.target_count:,} "
        f"states={len(prepared.liked_states):,}"
    )
    return prepared, eligibility


def cache_metadata(stage: str, config: Config) -> dict[str, object]:
    return {
        "cache_version": CACHE_VERSION,
        "stage": stage,
        "positive_listen_threshold": base.LISTEN_POSITIVE_THRESHOLD,
        "profile_lookback_days": base.PROFILE_LOOKBACK_DAYS,
        "event_time_feature": "exp-negative-gap-over-60-days",
    }


def save_cache(
    path: Path, prepared: SequenceData, stage: str, config: Config
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        metadata=np.asarray(json.dumps(cache_metadata(stage, config), sort_keys=True)),
        **prepared.__dict__,
    )
    log(f"saved public-derived full-sequence cache: {path}")


def load_cache(path: Path, stage: str, config: Config) -> SequenceData:
    with np.load(path, allow_pickle=False) as artifact:
        actual = json.loads(str(artifact["metadata"].item()))
        expected = cache_metadata(stage, config)
        if actual != expected:
            raise RuntimeError(f"sequence cache differs: {actual} != {expected}")
        values = {
            field: artifact[field].copy()
            for field in SequenceData.__dataclass_fields__
        }
    prepared = SequenceData(**values)
    log(
        f"loaded full-sequence cache: tokens={len(prepared.sequence_items):,} "
        f"targets={prepared.target_count:,}"
    )
    return prepared


def load_or_prepare(
    public: Path,
    stage: str,
    config: Config,
    cache: Path | None,
) -> tuple[SequenceData, CandidateSet]:
    candidate_name = (
        "validation_eligible_candidates.npz"
        if stage == "validation"
        else "test_eligible_candidates.npz"
    )
    eligibility = CandidateSet.read(public / "data" / candidate_name)
    if cache is not None and cache.is_file():
        prepared = load_cache(cache, stage, config)
        if not np.array_equal(prepared.item_ids, eligibility.item_ids):
            raise RuntimeError("cached catalog differs from candidate artifact")
        return prepared, eligibility
    prepared, built_eligibility = prepare_data(public, stage, config)
    if cache is not None:
        save_cache(cache, prepared, stage, config)
    return prepared, built_eligibility


def build_item_correction(
    prepared: SequenceData,
    config: Config,
    target_starts: np.ndarray,
    target_counts: np.ndarray,
) -> torch.Tensor:
    """Estimate in-batch propensities from the selected training targets."""
    counts = np.zeros(len(prepared.item_ids), dtype=np.float64)
    for user_row, count in enumerate(target_counts):
        if not count:
            continue
        offset = int(prepared.offsets[user_row])
        target_start = int(target_starts[user_row])
        counts += np.bincount(
            prepared.sequence_items[
                offset
                + target_start
                + 1 : offset
                + target_start
                + int(count)
                + 1
            ].astype(np.int64),
            minlength=len(prepared.item_ids),
        )
    total = float(counts.sum())
    probabilities = (counts + 1.0) / (total + len(counts))
    raw = np.power(
        (1.0 / len(counts)) / probabilities,
        config.inbatch_correction_power,
    )
    normalization = float(np.sum(counts * raw) / max(total, 1.0))
    correction = np.clip(
        raw / max(normalization, 1.0e-12),
        config.inbatch_correction_min,
        config.inbatch_correction_max,
    ).astype(np.float32)
    mean_on_targets = float(np.sum(counts * correction) / max(total, 1.0))
    log(
        f"inbatch correction: min={correction.min():.3f} "
        f"mean_on_targets={mean_on_targets:.3f} max={correction.max():.3f}"
    )
    return torch.from_numpy(correction)


def build_known_positive_filter(
    prepared: SequenceData, training_counts: np.ndarray
) -> np.ndarray:
    """Pack each user's complete public-training positive-item set."""
    width = (len(prepared.item_ids) + 7) // 8
    packed = np.zeros((len(prepared.user_ids), width), dtype=np.uint8)
    retained = 0
    for user_row, target_count in enumerate(training_counts):
        if not target_count:
            continue
        offset = int(prepared.offsets[user_row])
        # T adjacent targets consume sequence positions 0 through T.  The
        # chronological BCE suffix starts at T + 1 and is intentionally absent.
        items = np.unique(
            prepared.sequence_items[
                offset : offset + int(target_count) + 1
            ].astype(np.int64)
        )
        byte_columns = items >> 3
        bit_values = np.left_shift(
            np.uint8(1), (items & 7).astype(np.uint8)
        )
        np.bitwise_or.at(packed[user_row], byte_columns, bit_values)
        retained += len(items)
    log(
        f"full-history false-negative filter: users={len(prepared.user_ids):,} "
        f"unique_user_items={retained:,} bytes={packed.nbytes:,}"
    )
    return packed


def invalid_full_history_negatives(
    negatives: np.ndarray,
    targets: np.ndarray,
    user_rows: np.ndarray,
    known_positive_packed: np.ndarray,
    liked_items: np.ndarray,
) -> np.ndarray:
    """Return row-level invalid negatives using the complete training prefix."""
    byte_columns = negatives >> 3
    bit_columns = negatives & 7
    known = known_positive_packed[user_rows[:, None], byte_columns]
    invalid = ((known >> bit_columns) & 1).astype(bool)
    invalid |= negatives == targets[:, None]
    invalid |= (negatives[:, :, None] == liked_items[:, None, :]).any(axis=2)
    for column in range(1, negatives.shape[1]):
        invalid[:, column] |= (
            negatives[:, column, None] == negatives[:, :column]
        ).any(axis=1)
    return invalid


def draw_full_history_filtered_negatives(
    targets: torch.Tensor,
    user_rows: np.ndarray,
    liked_items: torch.Tensor,
    known_positive_packed: np.ndarray,
    n_items: int,
    config: Config,
    item_correction: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw mixed negatives and reject all known public-training positives."""
    targets_cpu = targets.detach().cpu().numpy().astype(np.int64, copy=False)
    liked_cpu = liked_items.detach().cpu().numpy().astype(np.int64, copy=False)
    batch_size = len(targets_cpu)
    uniform = torch.randint(
        n_items,
        (batch_size, config.uniform_negatives),
        generator=generator,
        device="cpu",
    )
    inbatch = torch.empty((batch_size, 0), dtype=torch.long)
    corrected_count = 0
    if config.inbatch_negatives:
        unique_targets = torch.unique(targets.detach().cpu())
        corrected_count = min(
            config.inbatch_negatives, max(len(unique_targets) - 1, 0)
        )
        if corrected_count:
            priorities = torch.rand(
                (batch_size, len(unique_targets)), generator=generator
            )
            priorities.masked_fill_(
                unique_targets[None, :].eq(
                    torch.from_numpy(targets_cpu)[:, None]
                ),
                -1.0,
            )
            columns = torch.topk(
                priorities, k=corrected_count, dim=1
            ).indices
            inbatch = unique_targets[columns]
        if corrected_count < config.inbatch_negatives:
            filler = torch.randint(
                n_items,
                (batch_size, config.inbatch_negatives - corrected_count),
                generator=generator,
                device="cpu",
            )
            inbatch = torch.cat([inbatch, filler], dim=1)

    negatives = torch.cat([uniform, inbatch], dim=1).numpy()
    correction = np.ones_like(negatives, dtype=np.float32)
    if corrected_count:
        correction[
            :, config.uniform_negatives : config.uniform_negatives + corrected_count
        ] = item_correction[inbatch[:, :corrected_count]].numpy()
    for _ in range(32):
        invalid = invalid_full_history_negatives(
            negatives,
            targets_cpu,
            user_rows,
            known_positive_packed,
            liked_cpu,
        )
        if not np.any(invalid):
            return (
                torch.from_numpy(negatives).to(device),
                torch.from_numpy(correction).to(device),
            )
        replacements = torch.randint(
            n_items,
            (int(invalid.sum()),),
            generator=generator,
            device="cpu",
        ).numpy()
        negatives[invalid] = replacements
        # Rejected in-batch items become ordinary uniform draws.
        correction[invalid] = 1.0
    invalid = invalid_full_history_negatives(
        negatives,
        targets_cpu,
        user_rows,
        known_positive_packed,
        liked_cpu,
    )
    if np.any(invalid):
        raise RuntimeError("full-history negative resampling failed")
    return (
        torch.from_numpy(negatives).to(device),
        torch.from_numpy(correction).to(device),
    )


def draw_training_negatives(
    positive_items: torch.Tensor,
    known_history: torch.Tensor,
    liked_items: torch.Tensor,
    user_rows: np.ndarray,
    known_positive_packed: np.ndarray | None,
    prepared: SequenceData,
    config: Config,
    item_correction: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.negative_filter_scope == "full-training-history":
        if known_positive_packed is None:
            raise RuntimeError("full-history negative filter was not built")
        return draw_full_history_filtered_negatives(
            positive_items,
            user_rows,
            liked_items,
            known_positive_packed,
            len(prepared.item_ids),
            config,
            item_correction,
            generator,
            device,
        )
    return base.draw_filtered_mixed_negatives(
        positive_items,
        known_history,
        liked_items,
        len(prepared.item_ids),
        config.uniform_negatives,
        config.inbatch_negatives,
        item_correction,
        generator,
        device,
    )


def event_sequence(
    model: base.SequentialTwoTower,
    items: torch.Tensor,
    features: torch.Tensor,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.cat([model.context_embedding(items), features], dim=-1)
    projected = model.event_projection(inputs)
    return model.sequence_encoder(projected, hidden)


def pool_explicit_states(
    model: base.SequentialTwoTower,
    prepared: SequenceData,
    state_ids: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unique_states, inverse = np.unique(state_ids, return_inverse=True)
    liked_items = torch.from_numpy(
        prepared.liked_states[unique_states].astype(np.int64)
    ).to(device)
    disliked_items = torch.from_numpy(
        prepared.disliked_states[unique_states].astype(np.int64)
    ).to(device)
    liked_vectors = model._mean_context_channel(liked_items)
    disliked_vectors = model._mean_context_channel(disliked_items)
    inverse_rows = torch.from_numpy(inverse.astype(np.int64)).to(device)
    counts = prepared.state_counts[state_ids]
    return (
        liked_vectors[inverse_rows],
        disliked_vectors[inverse_rows],
        torch.from_numpy(counts.astype(np.float32)).to(device),
    )


def user_queries(
    model: base.SequentialTwoTower,
    sequence_vectors: torch.Tensor,
    liked_vectors: torch.Tensor,
    disliked_vectors: torch.Tensor,
    user_features: torch.Tensor,
) -> torch.Tensor:
    return model.user_tower(
        torch.cat(
            [sequence_vectors, liked_vectors, disliked_vectors, user_features],
            dim=1,
        )
    )


def candidate_vectors(
    model: base.SequentialTwoTower,
    positives: torch.Tensor,
    negatives: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_items = torch.cat([positives, negatives.reshape(-1)])
    unique_items, inverse = torch.unique(all_items, return_inverse=True)
    encoded = model.encode_items(unique_items)
    positive_vectors = encoded[inverse[: len(positives)]]
    negative_vectors = encoded[inverse[len(positives) :]].reshape(
        len(positives), negatives.shape[1], -1
    )
    return positive_vectors, negative_vectors


def history_windows(
    sequence: np.ndarray,
    start: int,
    count: int,
    width: int,
    padding_item: int,
) -> np.ndarray:
    base_start = max(0, start - width + 1)
    segment = sequence[base_start : start + count]
    padded = np.concatenate(
        [np.full(width - 1, padding_item, dtype=np.uint16), segment]
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    offset = start - base_start
    return windows[offset : offset + count].copy()


def corrected_negative_loss(
    logits: torch.Tensor,
    correction: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    raw = F.softplus(logits)
    total = config.uniform_negatives + config.inbatch_negatives
    result = torch.zeros(len(logits), device=logits.device)
    if config.uniform_negatives:
        result += config.uniform_negatives * raw[
            :, : config.uniform_negatives
        ].mean(dim=1)
    if config.inbatch_negatives:
        selected = slice(config.uniform_negatives, total)
        weights = correction[:, selected]
        result += config.inbatch_negatives * (
            (raw[:, selected] * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0e-6)
        )
    return result / total


def retained_target_window_starts(
    total_targets: np.ndarray,
    validation_counts: np.ndarray,
    training_counts: np.ndarray,
) -> np.ndarray:
    """Return the discarded prefix before the retained chronological suffix."""
    training_region_ends = total_targets - validation_counts
    window_starts = training_region_ends - training_counts
    if (
        np.any(window_starts < 0)
        or np.any(training_counts < 0)
        or np.any(validation_counts < 0)
    ):
        raise RuntimeError("chronological target partition produced negative counts")
    if not np.array_equal(
        window_starts + training_counts + validation_counts, total_targets
    ):
        raise RuntimeError("chronological target window is not a final contiguous suffix")
    return window_starts


def chronological_target_partition(
    prepared: SequenceData,
    validation_fraction: float,
    max_train_targets_per_user: int,
    max_validation_targets_per_user: int,
) -> TargetPartition:
    """Select a recent contiguous window with a final out-of-time BCE suffix."""
    totals = np.maximum(prepared.lengths - 1, 0).astype(np.int64)
    validation = np.floor(totals * validation_fraction).astype(np.int64)
    validation[(totals >= 2) & (validation == 0)] = 1
    if max_validation_targets_per_user:
        validation = np.minimum(validation, max_validation_targets_per_user)

    history_training = totals - validation
    training = history_training.copy()
    if max_train_targets_per_user:
        training = np.minimum(training, max_train_targets_per_user)
    window_starts = retained_target_window_starts(totals, validation, training)
    return TargetPartition(
        window_starts=window_starts,
        training_counts=training,
        validation_counts=validation,
        history_training_counts=history_training,
    )


def load_organic_like_outcomes(
    public: Path, stage: str, item_ids: np.ndarray
) -> dict[int, dict[int, np.ndarray]]:
    """Load mapped organic-like timestamps from public fitting history."""
    prefixes = ("train",) if stage == "validation" else ("train", "val")
    frames = [
        pd.read_parquet(
            public / "data" / f"{prefix}_likes.parquet",
            columns=["uid", "timestamp", "item_id", "is_organic"],
        )
        for prefix in prefixes
    ]
    likes = pd.concat(frames, ignore_index=True)
    likes = likes[likes.is_organic == 1].copy()
    mapped = pd.Index(item_ids).get_indexer(
        likes.item_id.to_numpy(dtype=np.int64)
    )
    likes = likes.loc[mapped >= 0, ["uid", "timestamp"]].copy()
    likes["item"] = mapped[mapped >= 0].astype(np.int32, copy=False)
    likes = likes.sort_values(["uid", "item", "timestamp"], kind="stable")
    nested: dict[int, dict[int, list[int]]] = {}
    for raw_uid, raw_timestamp, raw_item in likes.itertuples(
        index=False, name=None
    ):
        nested.setdefault(int(raw_uid), {}).setdefault(int(raw_item), []).append(
            int(raw_timestamp)
        )
    result = {
        uid: {
            item: np.asarray(timestamps, dtype=np.int64)
            for item, timestamps in items.items()
        }
        for uid, items in nested.items()
    }
    log(
        f"loaded organic-like outcomes: rows={len(likes):,} "
        f"users={len(result):,}"
    )
    return result


def build_target_weights(
    public: Path,
    stage: str,
    prepared: SequenceData,
    training_counts: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Build nonzero per-token weights without dropping adjacent targets."""
    if config.target_weighting == "uniform":
        weights = np.ones(len(prepared.sequence_items), dtype=np.float32)
    else:
        weights = np.full(
            len(prepared.sequence_items),
            config.ordinary_target_weight,
            dtype=np.float32,
        )
        organic_strong = (
            (prepared.event_features[:, 0] > 0.5)
            & (prepared.event_features[:, 1] >= 0.8)
        )
        weights[organic_strong] = config.organic_strong_target_weight
        if config.target_weighting == "discovery-graded":
            like_outcomes = load_organic_like_outcomes(
                public, stage, prepared.item_ids
            )
            history_end = (
                base.TRAIN_END if stage == "validation" else base.VALIDATION_END
            )
            horizon = config.discovery_horizon_days * base.DAY
            grade_three = 0
            for user_row, uid in enumerate(prepared.user_ids):
                start = int(prepared.offsets[user_row])
                end = int(prepared.offsets[user_row + 1])
                if end - start < 2:
                    continue
                last_training_target = start + int(training_counts[user_row])
                first_validation_target = last_training_target + 1
                split_timestamp = (
                    int(prepared.sequence_timestamps[first_validation_target])
                    if first_validation_target < end
                    else history_end
                )
                user_likes = like_outcomes.get(int(uid), {})
                seen = {int(prepared.sequence_items[start])}
                for position in range(start + 1, end):
                    item = int(prepared.sequence_items[position])
                    novel = item not in seen
                    seen.add(item)
                    if not novel:
                        # Repeats remain training examples, but they cannot be
                        # eligible discovery outcomes at the corresponding
                        # cutoff and therefore keep only the ordinary weight.
                        weights[position] = config.ordinary_target_weight
                        continue
                    timestamps = user_likes.get(item)
                    if timestamps is None:
                        continue
                    timestamp = int(prepared.sequence_timestamps[position])
                    upper = min(
                        timestamp + horizon,
                        split_timestamp
                        if position <= last_training_target
                        else history_end,
                    )
                    outcome = int(np.searchsorted(timestamps, timestamp, side="left"))
                    if outcome < len(timestamps) and int(timestamps[outcome]) < upper:
                        weights[position] = config.organic_like_target_weight
                        grade_three += 1
            log(
                f"grade-3 first-encounter targets={grade_three:,} "
                f"horizon_days={config.discovery_horizon_days}"
            )
    log(
        f"target weighting={config.target_weighting} "
        f"min={weights.min():.3f} mean={weights.mean():.3f} "
        f"max={weights.max():.3f}"
    )
    return weights


def build_dense_block(
    prepared: SequenceData,
    target_weights: np.ndarray,
    users: np.ndarray,
    window_starts: np.ndarray,
    process_counts: np.ndarray,
    score_starts: np.ndarray,
    start: int,
    steps: int,
    config: Config,
) -> tuple[np.ndarray, ...]:
    """Build one causal sequence block and mark the target rows to score."""
    batch_size = len(users)
    padding_item = len(prepared.item_ids)
    input_items = np.full((batch_size, steps), padding_item, dtype=np.uint16)
    event_features = np.zeros((batch_size, steps, 3), dtype=np.float16)
    activity_features = np.zeros((batch_size, steps, 4), dtype=np.float16)
    state_ids = np.zeros((batch_size, steps), dtype=np.uint32)
    targets = np.zeros((batch_size, steps), dtype=np.uint16)
    weights = np.ones((batch_size, steps), dtype=np.float32)
    scored = np.zeros((batch_size, steps), dtype=bool)
    histories = np.full(
        (batch_size, steps, config.bptt_steps),
        padding_item,
        dtype=np.uint16,
    )
    processed = np.zeros(batch_size, dtype=np.int32)

    lengths = prepared.lengths
    for row, user_row in enumerate(users):
        count = min(steps, max(int(process_counts[row]) - start, 0))
        if not count:
            continue
        offset = int(prepared.offsets[user_row])
        window_start = int(window_starts[row])
        absolute_start = window_start + start
        positions = slice(
            offset + absolute_start,
            offset + absolute_start + count,
        )
        target_positions = slice(
            offset + absolute_start + 1,
            offset + absolute_start + count + 1,
        )
        input_items[row, :count] = prepared.sequence_items[positions]
        event_features[row, :count] = prepared.event_features[positions]
        activity_features[row, :count] = prepared.activity_features[positions]
        state_ids[row, :count] = prepared.state_ids[positions]
        targets[row, :count] = prepared.sequence_items[target_positions]
        weights[row, :count] = target_weights[target_positions]
        score_from = min(count, max(int(score_starts[row]) - start, 0))
        scored[row, score_from:count] = True
        processed[row] = count
        user_sequence = prepared.sequence_items[
            offset : offset + int(lengths[user_row])
        ]
        histories[row, :count] = history_windows(
            user_sequence,
            absolute_start,
            count,
            config.bptt_steps,
            padding_item,
        )
    return (
        input_items,
        event_features,
        activity_features,
        state_ids,
        targets,
        scored,
        histories,
        processed,
        weights,
    )


def score_target_microbatches(
    model: base.SequentialTwoTower,
    prepared: SequenceData,
    config: Config,
    sequence_vectors: torch.Tensor,
    flat_states: np.ndarray,
    flat_activity: np.ndarray,
    positive_items: torch.Tensor,
    negatives: torch.Tensor,
    negative_weights: torch.Tensor,
    sample_weights: np.ndarray,
    device: torch.device,
    backward: bool,
) -> tuple[dict[str, float], torch.Tensor | None]:
    """Compute one logical block's exact BCE in bounded-memory slices."""
    target_count = len(positive_items)
    total_weight = float(sample_weights.sum())
    if total_weight <= 0:
        raise RuntimeError("target block has no positive sample weight")
    sequence_gradient = torch.zeros_like(sequence_vectors) if backward else None
    statistics = {
        "loss_sum": 0.0,
        "positive_sum": 0.0,
        "negative_sum": 0.0,
        "pair_correct": 0.0,
        "weight_sum": 0.0,
    }
    for begin in range(0, target_count, config.loss_microbatch_size):
        end = min(begin + config.loss_microbatch_size, target_count)
        selected_states = flat_states[begin:end]
        if backward:
            sequence_input = (
                sequence_vectors[begin:end].detach().requires_grad_(True)
            )
        else:
            sequence_input = sequence_vectors[begin:end]
        liked_vectors, disliked_vectors, _ = pool_explicit_states(
            model, prepared, selected_states, device
        )
        counts = prepared.state_counts[selected_states]
        numeric = np.zeros((end - begin, 6), dtype=np.float32)
        numeric[:, :4] = flat_activity[begin:end]
        numeric[:, 4] = np.log1p(counts[:, 0]) / 6.0
        numeric[:, 5] = np.log1p(counts[:, 1]) / 4.0
        query_vectors = user_queries(
            model,
            sequence_input,
            liked_vectors,
            disliked_vectors,
            torch.from_numpy(numeric).to(device),
        )
        positive_vectors, negative_vectors = candidate_vectors(
            model,
            positive_items[begin:end],
            negatives[begin:end],
        )
        positive_logits = (query_vectors * positive_vectors).sum(dim=1)
        negative_logits = torch.einsum(
            "bd,bkd->bk", query_vectors, negative_vectors
        )
        positive_loss = F.softplus(-positive_logits)
        negative_loss = corrected_negative_loss(
            negative_logits, negative_weights[begin:end], config
        )
        selected_weights = torch.from_numpy(
            sample_weights[begin:end].astype(np.float32, copy=False)
        ).to(device)
        loss_sum = (
            selected_weights * (positive_loss + negative_loss) / 2.0
        ).sum()
        if not bool(torch.isfinite(loss_sum)):
            raise RuntimeError("dense next-item training produced non-finite loss")
        if backward:
            (loss_sum / total_weight).backward()
            if sequence_input.grad is None:
                raise RuntimeError("sequence microbatch did not produce a gradient")
            sequence_gradient[begin:end] = sequence_input.grad
        statistics["loss_sum"] += float(loss_sum.detach().cpu())
        statistics["positive_sum"] += float(positive_logits.detach().sum().cpu())
        statistics["negative_sum"] += float(negative_logits.detach().sum().cpu())
        statistics["pair_correct"] += float(
            (positive_logits.detach()[:, None] > negative_logits.detach()).sum().cpu()
        )
        statistics["weight_sum"] += float(selected_weights.sum().detach().cpu())
    return statistics, sequence_gradient


def release_accelerator_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()


def run_training_epoch(
    model: base.SequentialTwoTower,
    optimizer: torch.optim.Optimizer,
    prepared: SequenceData,
    config: Config,
    target_weights: np.ndarray,
    window_starts: np.ndarray,
    target_counts: np.ndarray,
    item_correction: torch.Tensor,
    known_positive_packed: np.ndarray | None,
    generator: torch.Generator,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    train_users = np.flatnonzero(target_counts > 0)
    sorted_users = train_users[
        np.argsort(target_counts[train_users], kind="stable")
    ]
    groups = [
        sorted_users[start : start + config.user_batch_size]
        for start in range(0, len(sorted_users), config.user_batch_size)
    ]
    group_order = torch.randperm(len(groups), generator=generator).numpy()
    negative_count = config.uniform_negatives + config.inbatch_negatives
    total_loss = 0.0
    total_targets = 0
    total_weight = 0.0
    total_positive = 0.0
    total_negative = 0.0
    total_pair_correct = 0.0
    batches = 0
    started = time.monotonic()

    for group_number in group_order:
        users = groups[int(group_number)]
        group_counts = target_counts[users]
        hidden = torch.zeros(
            1, len(users), config.embedding_dim, device=device
        )
        max_targets = int(group_counts.max())
        for start in range(0, max_targets, config.bptt_steps):
            steps = min(config.bptt_steps, max_targets - start)
            block = build_dense_block(
                prepared,
                target_weights,
                users,
                window_starts[users],
                group_counts,
                np.zeros(len(users), dtype=np.int64),
                start,
                steps,
                config,
            )
            sequence_outputs, _ = event_sequence(
                model,
                torch.from_numpy(block[0].astype(np.int64)).to(device),
                torch.from_numpy(block[1].astype(np.float32)).to(device),
                hidden,
            )
            hidden = torch.stack(
                [
                    sequence_outputs[row, count - 1]
                    if count
                    else hidden[0, row]
                    for row, count in enumerate(block[7])
                ]
            ).unsqueeze(0).detach()
            flat_scored = block[5].reshape(-1)
            if not np.any(flat_scored):
                continue
            mask = torch.from_numpy(flat_scored).to(device)
            sequence_vectors = sequence_outputs.reshape(
                -1, config.embedding_dim
            )[mask]
            flat_states = block[3].reshape(-1)[flat_scored]
            positive_items = torch.from_numpy(
                block[4].reshape(-1)[flat_scored].astype(np.int64)
            ).to(device)
            known_history = torch.from_numpy(
                block[6].reshape(-1, config.bptt_steps)[flat_scored].astype(
                    np.int64
                )
            ).to(device)
            liked_items = torch.from_numpy(
                prepared.liked_states[flat_states].astype(np.int64)
            ).to(device)
            flat_user_rows = np.broadcast_to(
                users[:, None], block[5].shape
            ).reshape(-1)[flat_scored]
            negatives, negative_weights = draw_training_negatives(
                positive_items,
                known_history,
                liked_items,
                flat_user_rows,
                known_positive_packed,
                prepared,
                config,
                item_correction,
                generator,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            statistics, sequence_gradient = score_target_microbatches(
                model,
                prepared,
                config,
                sequence_vectors,
                flat_states,
                block[2].reshape(-1, 4)[flat_scored].astype(np.float32),
                positive_items,
                negatives,
                negative_weights,
                block[8].reshape(-1)[flat_scored],
                device,
                backward=True,
            )
            if sequence_gradient is None:
                raise RuntimeError("training block omitted its sequence gradient")
            sequence_vectors.backward(sequence_gradient)
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

            count = len(positive_items)
            total_loss += statistics["loss_sum"]
            total_targets += count
            total_weight += statistics["weight_sum"]
            total_positive += statistics["positive_sum"]
            total_negative += statistics["negative_sum"]
            total_pair_correct += statistics["pair_correct"]
            batches += 1
            if batches % 25 == 0:
                release_accelerator_cache(device)
            if batches % 250 == 0:
                batch_bce = statistics["loss_sum"] / max(
                    statistics["weight_sum"], 1.0e-12
                )
                running_bce = total_loss / max(total_weight, 1.0e-12)
                log(
                    f"epoch={epoch}/{config.epochs} batches={batches:,} "
                    f"targets={total_targets:,} "
                    f"batch_bce={batch_bce:.6f} "
                    f"running_bce={running_bce:.6f} "
                    f"elapsed={time.monotonic() - started:.1f}s"
                )

    return {
        "epoch": float(epoch),
        "loss": total_loss / max(total_weight, 1.0e-12),
        "positive_logit": total_positive / max(total_targets, 1),
        "negative_logit": total_negative / max(total_targets * negative_count, 1),
        "pair_accuracy": total_pair_correct
        / max(total_targets * negative_count, 1),
        "targets": float(total_targets),
        "target_weight_sum": total_weight,
        "batches": float(batches),
        "seconds": time.monotonic() - started,
    }


@torch.no_grad()
def evaluate_validation_bce(
    model: base.SequentialTwoTower,
    prepared: SequenceData,
    config: Config,
    target_weights: np.ndarray,
    window_starts: np.ndarray,
    training_counts: np.ndarray,
    validation_counts: np.ndarray,
    item_correction: torch.Tensor,
    known_positive_packed: np.ndarray | None,
    device: torch.device,
) -> dict[str, float]:
    """Measure deterministic BCE on each user's chronological target suffix."""
    model.eval()
    totals = training_counts + validation_counts
    users_with_validation = np.flatnonzero(validation_counts > 0)
    ordered = users_with_validation[
        np.argsort(totals[users_with_validation], kind="stable")
    ]
    groups = [
        ordered[start : start + config.user_batch_size]
        for start in range(0, len(ordered), config.user_batch_size)
    ]
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 20_000)
    total_loss = 0.0
    total_targets = 0
    total_weight = 0.0
    started = time.monotonic()

    for group_number, users in enumerate(groups, start=1):
        group_totals = totals[users]
        group_starts = training_counts[users]
        hidden = torch.zeros(
            1, len(users), config.embedding_dim, device=device
        )
        max_targets = int(group_totals.max())
        for start in range(0, max_targets, config.bptt_steps):
            steps = min(config.bptt_steps, max_targets - start)
            block = build_dense_block(
                prepared,
                target_weights,
                users,
                window_starts[users],
                group_totals,
                group_starts,
                start,
                steps,
                config,
            )
            sequence_outputs, _ = event_sequence(
                model,
                torch.from_numpy(block[0].astype(np.int64)).to(device),
                torch.from_numpy(block[1].astype(np.float32)).to(device),
                hidden,
            )
            hidden = torch.stack(
                [
                    sequence_outputs[row, count - 1]
                    if count
                    else hidden[0, row]
                    for row, count in enumerate(block[7])
                ]
            ).unsqueeze(0)
            flat_scored = block[5].reshape(-1)
            if not np.any(flat_scored):
                continue
            mask = torch.from_numpy(flat_scored).to(device)
            sequence_vectors = sequence_outputs.reshape(
                -1, config.embedding_dim
            )[mask]
            flat_states = block[3].reshape(-1)[flat_scored]
            positive_items = torch.from_numpy(
                block[4].reshape(-1)[flat_scored].astype(np.int64)
            ).to(device)
            known_history = torch.from_numpy(
                block[6].reshape(-1, config.bptt_steps)[flat_scored].astype(
                    np.int64
                )
            ).to(device)
            liked_items = torch.from_numpy(
                prepared.liked_states[flat_states].astype(np.int64)
            ).to(device)
            flat_user_rows = np.broadcast_to(
                users[:, None], block[5].shape
            ).reshape(-1)[flat_scored]
            negatives, negative_weights = draw_training_negatives(
                positive_items,
                known_history,
                liked_items,
                flat_user_rows,
                known_positive_packed,
                prepared,
                config,
                item_correction,
                generator,
                device,
            )
            statistics, _ = score_target_microbatches(
                model,
                prepared,
                config,
                sequence_vectors,
                flat_states,
                block[2].reshape(-1, 4)[flat_scored].astype(np.float32),
                positive_items,
                negatives,
                negative_weights,
                block[8].reshape(-1)[flat_scored],
                device,
                backward=False,
            )
            total_loss += statistics["loss_sum"]
            total_targets += len(positive_items)
            total_weight += statistics["weight_sum"]
        if group_number % 25 == 0:
            release_accelerator_cache(device)
    return {
        "validation_loss": total_loss / max(total_weight, 1.0e-12),
        "validation_targets": float(total_targets),
        "validation_target_weight_sum": total_weight,
        "validation_seconds": time.monotonic() - started,
    }


def snapshot_model(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def save_epoch_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model_state: dict[str, torch.Tensor],
    config: Config,
    diagnostics: dict[str, float],
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"epoch-{epoch:02d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "config": asdict(config),
            "diagnostics": diagnostics,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
        },
        path,
    )
    log(f"saved epoch checkpoint: {path}")
    return path


def train_model(
    public: Path,
    stage: str,
    prepared: SequenceData,
    config: Config,
    device: torch.device,
    checkpoint_dir: Path | None,
    resume_checkpoint: Path | None,
) -> tuple[base.SequentialTwoTower, list[dict[str, float]]]:
    partition = chronological_target_partition(
        prepared,
        config.validation_fraction,
        config.max_train_targets_per_user,
        config.max_validation_targets_per_user,
    )
    training_counts = partition.training_counts
    validation_counts = partition.validation_counts
    log(
        f"chronological BCE window: train={int(training_counts.sum()):,} "
        f"validation={int(validation_counts.sum()):,} "
        f"selected={partition.selected_count:,} "
        f"discarded_old_targets={int(partition.window_starts.sum()):,} "
        f"all={prepared.target_count:,}"
    )
    model = base.SequentialTwoTower(prepared.item_features, model_config(config)).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 10_000)
    item_correction = build_item_correction(
        prepared,
        config,
        partition.window_starts,
        training_counts,
    )
    target_weights = build_target_weights(
        public,
        stage,
        prepared,
        partition.history_training_counts,
        config,
    )
    known_positive_packed = (
        build_known_positive_filter(prepared, partition.history_training_counts)
        if config.negative_filter_scope == "full-training-history"
        else None
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    start_epoch = 0

    if resume_checkpoint is not None:
        payload = torch.load(
            resume_checkpoint, map_location=device, weights_only=False
        )
        stored_config = payload.get("config", {})
        architecture_fields = (
            "bptt_steps",
            "embedding_dim",
            "hidden_dim",
            "user_hidden_1",
            "user_hidden_2",
            "item_hidden_1",
            "item_hidden_2",
        )
        mismatches = {
            name: (stored_config.get(name), getattr(config, name))
            for name in architecture_fields
            if stored_config.get(name) != getattr(config, name)
        }
        if mismatches:
            raise RuntimeError(
                f"resume checkpoint architecture differs: {mismatches}"
            )
        stored_filter_scope = stored_config.get(
            "negative_filter_scope", "last-100"
        )
        stored_target_weighting = stored_config.get(
            "target_weighting", "uniform"
        )
        stored_weight_values = (
            stored_config.get("ordinary_target_weight", 1.0),
            stored_config.get("organic_strong_target_weight", 1.0),
            stored_config.get("organic_like_target_weight", 3.0),
            stored_config.get("discovery_horizon_days", 7),
        )
        current_weight_values = (
            config.ordinary_target_weight,
            config.organic_strong_target_weight,
            config.organic_like_target_weight,
            config.discovery_horizon_days,
        )
        stored_window_values = (
            stored_config.get("validation_fraction", 0.10),
            stored_config.get("max_train_targets_per_user", 0),
            stored_config.get("max_validation_targets_per_user", 0),
        )
        current_window_values = (
            config.validation_fraction,
            config.max_train_targets_per_user,
            config.max_validation_targets_per_user,
        )
        objective_changed = (
            stored_filter_scope != config.negative_filter_scope
            or stored_target_weighting != config.target_weighting
            or stored_weight_values != current_weight_values
            or stored_window_values != current_window_values
        )
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload and not objective_changed:
            optimizer.load_state_dict(payload["optimizer_state"])
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate
                group["weight_decay"] = config.weight_decay
            log(f"restored optimizer state from {resume_checkpoint}")
        else:
            log(
                f"using a fresh AdamW optimizer after loading {resume_checkpoint}"
            )
        if "generator_state" in payload and not objective_changed:
            generator.set_state(payload["generator_state"])
        start_epoch = int(payload["epoch"])
        if start_epoch >= config.epochs:
            raise RuntimeError(
                f"resume epoch {start_epoch} must be below --epochs {config.epochs}"
            )
        prior = {
            str(name): float(value)
            for name, value in payload.get("diagnostics", {}).items()
        }
        if objective_changed:
            prior.pop("validation_loss", None)
            log(
                "BCE objective changed: filter "
                f"{stored_filter_scope}->{config.negative_filter_scope}, "
                "target weighting "
                f"{stored_target_weighting}->{config.target_weighting}, "
                f"target window {stored_window_values}->{current_window_values}; "
                "recomputing the resume checkpoint's validation BCE"
            )
        if "validation_loss" not in prior:
            prior.update(
                evaluate_validation_bce(
                    model,
                    prepared,
                    config,
                    target_weights,
                    partition.window_starts,
                    training_counts,
                    validation_counts,
                    item_correction,
                    known_positive_packed,
                    device,
                )
            )
        history.append(prior)
        best_loss = float(prior["validation_loss"])
        best_epoch = start_epoch
        best_state = snapshot_model(model)
        log(
            f"resumed validation-BCE training: epoch={start_epoch} "
            f"loss={best_loss:.6f}"
        )

    for epoch in range(start_epoch + 1, config.epochs + 1):
        diagnostics = run_training_epoch(
            model,
            optimizer,
            prepared,
            config,
            target_weights,
            partition.window_starts,
            training_counts,
            item_correction,
            known_positive_packed,
            generator,
            device,
            epoch,
        )
        validation = evaluate_validation_bce(
            model,
            prepared,
            config,
            target_weights,
            partition.window_starts,
            training_counts,
            validation_counts,
            item_correction,
            known_positive_packed,
            device,
        )
        diagnostics.update(validation)
        history.append(diagnostics)
        current_state = snapshot_model(model)
        if checkpoint_dir is not None:
            save_epoch_checkpoint(
                checkpoint_dir,
                epoch,
                current_state,
                config,
                diagnostics,
                optimizer,
                generator,
            )
        improved = (
            diagnostics["validation_loss"]
            < best_loss - config.early_stopping_min_delta
        )
        if improved:
            best_loss = diagnostics["validation_loss"]
            best_epoch = epoch
            best_state = current_state
            epochs_without_improvement = 0
            if checkpoint_dir is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "config": asdict(config),
                        "diagnostics": diagnostics,
                        "model_state": current_state,
                        "optimizer_state": optimizer.state_dict(),
                        "generator_state": generator.get_state(),
                    },
                    checkpoint_dir / "best.pt",
                )
        else:
            epochs_without_improvement += 1
        log(
            f"epoch={epoch}/{config.epochs} train_loss={diagnostics['loss']:.5f} "
            f"validation_loss={diagnostics['validation_loss']:.5f} "
            f"pair_accuracy={diagnostics['pair_accuracy']:.3f} "
            f"train_targets={int(diagnostics['targets']):,} "
            f"validation_targets={int(diagnostics['validation_targets']):,} "
            f"train_seconds={diagnostics['seconds']:.1f} "
            f"validation_seconds={diagnostics['validation_seconds']:.1f}"
        )
        release_accelerator_cache(device)
        if epochs_without_improvement >= config.early_stopping_patience:
            log(
                f"early stopping after epoch={epoch}; "
                f"best_epoch={best_epoch} best_validation_loss={best_loss:.6f}"
            )
            break

    if best_state is None:
        raise RuntimeError("training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    log(
        f"restored best validation-BCE checkpoint: epoch={best_epoch} "
        f"loss={best_loss:.6f}"
    )
    return model, history


@torch.inference_mode()
def inference_queries(
    model: base.SequentialTwoTower,
    prepared: SequenceData,
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    sequence_rows = prepared.inference_user_rows
    lengths = np.where(
        sequence_rows >= 0, prepared.lengths[np.maximum(sequence_rows, 0)], 0
    )
    ordered = np.argsort(lengths, kind="stable")
    result = torch.zeros(
        len(sequence_rows), config.embedding_dim, dtype=torch.float32
    )
    for group_start in range(0, len(ordered), config.user_batch_size):
        output_rows = ordered[group_start : group_start + config.user_batch_size]
        rows = sequence_rows[output_rows]
        group_lengths = lengths[output_rows]
        hidden = torch.zeros(
            1, len(output_rows), config.embedding_dim, device=device
        )
        for start in range(0, int(group_lengths.max(initial=0)), config.bptt_steps):
            steps = min(
                config.bptt_steps,
                int(group_lengths.max(initial=0)) - start,
            )
            items = np.full(
                (len(output_rows), steps),
                len(prepared.item_ids),
                dtype=np.uint16,
            )
            features = np.zeros((len(output_rows), steps, 3), dtype=np.float16)
            counts = np.zeros(len(output_rows), dtype=np.int32)
            for local_row, sequence_row in enumerate(rows):
                if sequence_row < 0:
                    continue
                count = min(steps, max(int(group_lengths[local_row]) - start, 0))
                if not count:
                    continue
                offset = int(prepared.offsets[sequence_row])
                selected = slice(offset + start, offset + start + count)
                items[local_row, :count] = prepared.sequence_items[selected]
                features[local_row, :count] = prepared.event_features[selected]
                counts[local_row] = count
            outputs, _ = event_sequence(
                model,
                torch.from_numpy(items.astype(np.int64)).to(device),
                torch.from_numpy(features.astype(np.float32)).to(device),
                hidden,
            )
            next_rows: list[torch.Tensor] = []
            for local_row, count in enumerate(counts):
                next_rows.append(
                    outputs[local_row, count - 1]
                    if count
                    else hidden[0, local_row]
                )
            hidden = torch.stack(next_rows).unsqueeze(0)

        liked_items = torch.from_numpy(
            prepared.inference_liked_items[output_rows].astype(np.int64)
        ).to(device)
        disliked_items = torch.from_numpy(
            prepared.inference_disliked_items[output_rows].astype(np.int64)
        ).to(device)
        queries = user_queries(
            model,
            hidden.squeeze(0),
            model._mean_context_channel(liked_items),
            model._mean_context_channel(disliked_items),
            torch.from_numpy(
                prepared.inference_user_features[output_rows].astype(np.float32)
            ).to(device),
        )
        result[torch.from_numpy(output_rows.astype(np.int64))] = queries.cpu()
    return result


@torch.inference_mode()
def rank_candidates(
    model: base.SequentialTwoTower,
    prepared: SequenceData,
    eligibility: CandidateSet,
    config: Config,
    device: torch.device,
    ranking_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    queries = inference_queries(model, prepared, config, device)
    item_parts: list[torch.Tensor] = []
    for start in range(0, len(prepared.item_ids), 2_048):
        rows = torch.arange(
            start,
            min(start + 2_048, len(prepared.item_ids)),
            dtype=torch.long,
            device=device,
        )
        item_parts.append(model.encode_items(rows).cpu())
    items = torch.cat(item_parts)
    rows: list[tuple[int, int, int]] = []
    for start in range(0, len(eligibility.uids), ranking_batch_size):
        end = min(start + ranking_batch_size, len(eligibility.uids))
        scores = queries[start:end] @ items.T
        eligible = np.unpackbits(
            eligibility.packed[start:end],
            axis=1,
            count=len(prepared.item_ids),
            bitorder="little",
        ).astype(bool, copy=False)
        scores.masked_fill_(~torch.from_numpy(eligible), -torch.inf)
        selected = torch.topk(scores, k=base.TOP_N, dim=1).indices.numpy()
        for local_row, columns in enumerate(selected):
            uid = int(eligibility.uids[start + local_row])
            rows.extend(
                (uid, int(prepared.item_ids[column]), rank)
                for rank, column in enumerate(columns, start=1)
            )
        if end == len(eligibility.uids) or end % 1_024 == 0:
            log(f"ranked users={end:,}/{len(eligibility.uids):,}")
    query_norm = torch.linalg.vector_norm(queries, dim=1).numpy()
    item_norm = torch.linalg.vector_norm(items, dim=1).numpy()
    diagnostics = {
        "query_norm_mean": float(query_norm.mean()),
        "query_norm_std": float(query_norm.std()),
        "item_norm_mean": float(item_norm.mean()),
        "item_norm_std": float(item_norm.std()),
    }
    return pd.DataFrame(rows, columns=["uid", "item_id", "rank"]), diagnostics


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
        "--stage", choices=("validation", "submission"), default="validation"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--prepared-cache", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bptt-steps", type=int, default=Config.bptt_steps)
    parser.add_argument(
        "--user-batch-size", type=int, default=Config.user_batch_size
    )
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument(
        "--loss-microbatch-size",
        type=int,
        default=Config.loss_microbatch_size,
    )
    parser.add_argument(
        "--negative-filter-scope",
        choices=("full-training-history", "last-100"),
        default=Config.negative_filter_scope,
    )
    parser.add_argument(
        "--target-weighting",
        choices=("discovery-graded", "organic-strong", "uniform"),
        default=Config.target_weighting,
    )
    parser.add_argument(
        "--ordinary-target-weight",
        type=float,
        default=Config.ordinary_target_weight,
    )
    parser.add_argument(
        "--organic-strong-target-weight",
        type=float,
        default=Config.organic_strong_target_weight,
    )
    parser.add_argument(
        "--organic-like-target-weight",
        type=float,
        default=Config.organic_like_target_weight,
    )
    parser.add_argument(
        "--discovery-horizon-days",
        type=int,
        default=Config.discovery_horizon_days,
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=Config.validation_fraction,
    )
    parser.add_argument(
        "--max-train-targets-per-user",
        type=int,
        default=Config.max_train_targets_per_user,
        help="retain this many recent pre-validation targets per user; 0 is uncapped",
    )
    parser.add_argument(
        "--max-validation-targets-per-user",
        type=int,
        default=Config.max_validation_targets_per_user,
        help="retain this many final validation targets per user; 0 is uncapped",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=Config.early_stopping_patience,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=Config.early_stopping_min_delta,
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--ranking-batch-size", type=int, default=64)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    if args.threads < 1 or args.bptt_steps < 2 or args.user_batch_size < 1:
        parser.error("threads, BPTT steps, and user batch size must be positive")
    if (
        args.epochs < 1
        or args.learning_rate <= 0
        or args.loss_microbatch_size < 1
        or args.early_stopping_patience < 1
        or args.early_stopping_min_delta < 0
        or args.ordinary_target_weight <= 0
        or args.organic_strong_target_weight <= 0
        or args.organic_like_target_weight <= 0
        or args.discovery_horizon_days < 1
        or args.max_train_targets_per_user < 0
        or args.max_validation_targets_per_user < 0
    ):
        parser.error("training counts, learning rate, and patience are invalid")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation fraction must be strictly between zero and one")
    public = args.public_dir.resolve()
    if not (public / "data" / "candidate_catalog.parquet").is_file():
        parser.error(f"public task data not found under {public}")
    config = Config(
        bptt_steps=args.bptt_steps,
        user_batch_size=args.user_batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        loss_microbatch_size=args.loss_microbatch_size,
        negative_filter_scope=args.negative_filter_scope,
        target_weighting=args.target_weighting,
        ordinary_target_weight=args.ordinary_target_weight,
        organic_strong_target_weight=args.organic_strong_target_weight,
        organic_like_target_weight=args.organic_like_target_weight,
        discovery_horizon_days=args.discovery_horizon_days,
        validation_fraction=args.validation_fraction,
        max_train_targets_per_user=args.max_train_targets_per_user,
        max_validation_targets_per_user=args.max_validation_targets_per_user,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        seed=args.seed,
    )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    seed_everything(config.seed)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")

    started = time.monotonic()
    prepared, eligibility = load_or_prepare(
        public,
        args.stage,
        config,
        args.prepared_cache.resolve() if args.prepared_cache else None,
    )
    target_partition = chronological_target_partition(
        prepared,
        config.validation_fraction,
        config.max_train_targets_per_user,
        config.max_validation_targets_per_user,
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "sequence_users": len(prepared.user_ids),
                    "sequence_tokens": len(prepared.sequence_items),
                    "next_item_targets": prepared.target_count,
                    "selected_training_targets": int(
                        target_partition.training_counts.sum()
                    ),
                    "selected_validation_targets": int(
                        target_partition.validation_counts.sum()
                    ),
                    "discarded_old_targets": int(
                        target_partition.window_starts.sum()
                    ),
                    "target_users": len(eligibility.uids),
                    "runtime_seconds": round(time.monotonic() - started, 3),
                    "config": asdict(config),
                },
                indent=2,
            )
        )
        return 0

    model, training_history = train_model(
        public,
        args.stage,
        prepared,
        config,
        device,
        args.checkpoint_dir.resolve() if args.checkpoint_dir else None,
        args.resume_checkpoint.resolve() if args.resume_checkpoint else None,
    )
    ranking, embedding_diagnostics = rank_candidates(
        model,
        prepared,
        eligibility,
        config,
        device,
        args.ranking_batch_size,
    )
    write_ranking(ranking, args.output.resolve())
    print(
        json.dumps(
            {
                "baseline": "yambda-windowed-sequence-two-tower-base",
                "stage": args.stage,
                "output": str(args.output.resolve()),
                "rows": len(ranking),
                "users": int(ranking.uid.nunique()),
                "runtime_seconds": round(time.monotonic() - started, 3),
                "public_inputs_only": True,
                "architecture": {
                    "family": "pure-windowed-sequence-two-tower",
                    "sequence_training": (
                        "every-adjacent-target-within-recent-chronological-window"
                    ),
                    "sequence_encoder": "one-layer-GRU-dense-timestep-loss",
                    "bptt_hidden_state": "carried-and-detached",
                    "model_selection": "chronological-validation-BCE-early-stopping",
                    "loss_execution": "gradient-accumulated-microbatches",
                    "user_id_embedding": False,
                    "user_mlp": "582-512-256-192",
                    "item_mlp": "196-512-256-192",
                    "normalization": "none",
                    "score": "raw-dot-product",
                    "objective": "sample-weighted-BCE-at-every-timestep",
                    "negative_sampling": (
                        "24-uniform-plus-8-propensity-corrected-inbatch"
                    ),
                    "false_negative_filter": config.negative_filter_scope,
                    "target_weighting": config.target_weighting,
                },
                "config": asdict(config),
                "sequence_tokens": len(prepared.sequence_items),
                "all_next_item_targets": prepared.target_count,
                "selected_training_targets": int(
                    target_partition.training_counts.sum()
                ),
                "selected_validation_targets": int(
                    target_partition.validation_counts.sum()
                ),
                "discarded_old_targets": int(target_partition.window_starts.sum()),
                "training_history": training_history,
                "embedding_diagnostics": embedding_diagnostics,
                "torch_version": torch.__version__,
                "device": str(device),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
