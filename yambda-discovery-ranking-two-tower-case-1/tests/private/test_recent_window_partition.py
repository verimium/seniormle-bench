#!/usr/bin/env python3
"""Synthetic regression test for the recent-window anchoring invariant."""

from __future__ import annotations

import unittest

import baseline_solver
import numpy as np
from reference_solution import reference_solver


class SyntheticPrepared:
    @property
    def lengths(self) -> np.ndarray:
        # Adjacent-target totals are 1, 5, and 150.
        return np.asarray([2, 6, 151], dtype=np.int64)


class RecentWindowPartitionTest(unittest.TestCase):
    def partition(self, solver: object) -> object:
        return solver.chronological_target_partition(
            SyntheticPrepared(),
            validation_fraction=0.10,
            max_train_targets_per_user=4,
            max_validation_targets_per_user=2,
        )

    def assert_right_anchored(self, partition: object) -> None:
        totals = np.asarray([1, 5, 150], dtype=np.int64)
        expected_starts = partition.history_training_counts - partition.training_counts
        np.testing.assert_array_equal(partition.window_starts, expected_starts)
        np.testing.assert_array_equal(
            partition.window_starts
            + partition.training_counts
            + partition.validation_counts,
            totals,
        )

    def test_reference_solution_is_right_anchored(self) -> None:
        self.assert_right_anchored(self.partition(reference_solver))

    def test_planted_model_is_a_negative_control(self) -> None:
        partition = self.partition(baseline_solver)
        expected_starts = partition.history_training_counts - partition.training_counts
        self.assertFalse(np.array_equal(partition.window_starts, expected_starts))


if __name__ == "__main__":
    unittest.main()
