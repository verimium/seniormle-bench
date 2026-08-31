#!/usr/bin/env python3
"""Authoring checks for the production and shared-negative reference models."""

from __future__ import annotations

import unittest

import numpy as np
import production_solver
import torch
from reference_solution import reference_solver


class SyntheticPrepared:
    lengths = np.asarray([2, 6, 151, 1001], dtype=np.int64)


class SharedNegativePoolTest(unittest.TestCase):
    def test_production_model_already_uses_the_correct_recent_window(self) -> None:
        arguments = {
            "validation_fraction": 0.10,
            "max_train_targets_per_user": 4,
            "max_validation_targets_per_user": 2,
        }
        production = production_solver.chronological_target_partition(
            SyntheticPrepared(), **arguments
        )
        reference = reference_solver.chronological_target_partition(
            SyntheticPrepared(), **arguments
        )

        for field in (
            "window_starts",
            "training_counts",
            "validation_counts",
            "history_training_counts",
        ):
            np.testing.assert_array_equal(
                getattr(production, field), getattr(reference, field)
            )

    def test_only_reference_exposes_the_shared_negative_pool(self) -> None:
        self.assertFalse(hasattr(production_solver.Config, "shared_negative_pool_size"))
        self.assertFalse(
            hasattr(production_solver, "candidate_vectors_with_shared_pool")
        )
        self.assertEqual(reference_solver.Config.shared_negative_pool_size, 512)
        self.assertTrue(callable(reference_solver.candidate_vectors_with_shared_pool))

    def test_reference_filters_known_items_from_shared_negative_pool(self) -> None:
        packed = np.zeros((2, 2), dtype=np.uint8)
        packed[0, 0] |= np.uint8(1 << 2)
        packed[1, 0] |= np.uint8(1 << 4)

        class SharedNegativePrepared:
            liked_states = np.array([[3, 10], [2, 10]], dtype=np.uint16)

        valid = reference_solver.valid_shared_negatives(
            torch.tensor([1, 2, 3, 4]),
            torch.tensor([1, 9]),
            np.array([0, 1]),
            np.array([0, 1]),
            SharedNegativePrepared(),
            packed,
            torch.device("cpu"),
        )

        self.assertEqual(
            valid.tolist(),
            [
                [False, False, False, True],
                [True, False, True, False],
            ],
        )


if __name__ == "__main__":
    unittest.main()
