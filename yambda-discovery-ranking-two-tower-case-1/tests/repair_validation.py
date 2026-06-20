#!/usr/bin/env python3
"""Deterministically validate the repaired temporal partition in submitted code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

MODEL_ARTIFACT = "train_current_model.py"
RESULT_MARKER = "__YAMBDA_REPAIR_RESULT__="
DEFAULT_TIMEOUT_SECONDS = 45

SCENARIOS = (
    {
        "lengths": [2, 6, 151, 1001],
        "validation_fraction": 0.10,
        "max_train_targets_per_user": 4,
        "max_validation_targets_per_user": 2,
    },
    {
        "lengths": [1, 3, 8, 33, 129],
        "validation_fraction": 0.25,
        "max_train_targets_per_user": 7,
        "max_validation_targets_per_user": 3,
    },
    {
        "lengths": [1, 2, 4, 10, 65],
        "validation_fraction": 0.0,
        "max_train_targets_per_user": 2,
        "max_validation_targets_per_user": 0,
    },
)

RUNNER_SOURCE = r'''from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    source = Path(sys.argv[1])
    scenarios = json.load(sys.stdin)
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("submitted_current_model", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load submitted model: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    partition_function = getattr(module, "chronological_target_partition", None)
    if not callable(partition_function):
        raise RuntimeError(
            "submitted model must define chronological_target_partition"
        )

    results = []
    for scenario in scenarios:
        class SyntheticPrepared:
            lengths = np.asarray(scenario["lengths"], dtype=np.int64)

        partition = partition_function(
            SyntheticPrepared(),
            validation_fraction=float(scenario["validation_fraction"]),
            max_train_targets_per_user=int(
                scenario["max_train_targets_per_user"]
            ),
            max_validation_targets_per_user=int(
                scenario["max_validation_targets_per_user"]
            ),
        )
        results.append(
            {
                name: np.asarray(getattr(partition, name), dtype=np.int64).tolist()
                for name in (
                    "window_starts",
                    "training_counts",
                    "validation_counts",
                    "history_training_counts",
                )
            }
        )
    print("__YAMBDA_REPAIR_RESULT__=" + json.dumps(results, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


class RepairValidationError(ValueError):
    """Raised when the submitted implementation does not satisfy the repair."""


def _expected_partition(scenario: dict[str, Any]) -> dict[str, list[int]]:
    totals = [max(int(length) - 1, 0) for length in scenario["lengths"]]
    validation = [
        int(total * float(scenario["validation_fraction"])) for total in totals
    ]
    validation = [
        1 if total >= 2 and count == 0 else count
        for total, count in zip(totals, validation, strict=True)
    ]
    validation_cap = int(scenario["max_validation_targets_per_user"])
    if validation_cap:
        validation = [min(count, validation_cap) for count in validation]
    history_training = [
        total - count for total, count in zip(totals, validation, strict=True)
    ]
    training = list(history_training)
    training_cap = int(scenario["max_train_targets_per_user"])
    if training_cap:
        training = [min(count, training_cap) for count in training]
    window_starts = [
        history - count
        for history, count in zip(history_training, training, strict=True)
    ]
    return {
        "window_starts": window_starts,
        "training_counts": training,
        "validation_counts": validation,
        "history_training_counts": history_training,
    }


def _drop_privileges(uid: int, gid: int):
    def demote() -> None:
        if os.geteuid() == 0:
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)

    return demote


def validate_recent_window_repair(
    solution_dir: Path,
    *,
    run_uid: int | None = None,
    run_gid: int | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Run synthetic cases against the submitted model and enforce the invariant."""
    source = solution_dir / MODEL_ARTIFACT
    if source.is_symlink():
        raise RepairValidationError(f"{MODEL_ARTIFACT} must not be a symbolic link")
    if not source.is_file():
        raise RepairValidationError(
            f"missing {MODEL_ARTIFACT}; submit the repaired model implementation"
        )

    uid = os.getuid() if run_uid is None else run_uid
    gid = os.getgid() if run_gid is None else run_gid
    if uid <= 0 or gid <= 0:
        raise RuntimeError("repair-test UID and GID must be positive")
    if os.geteuid() != 0 and (uid != os.getuid() or gid != os.getgid()):
        raise RuntimeError("cannot run repair test as another user without root")

    with tempfile.TemporaryDirectory(prefix="yambda-repair-") as directory:
        scratch = Path(directory)
        runner = scratch / "runner.py"
        runner.write_text(RUNNER_SOURCE, encoding="utf-8")
        scratch.chmod(0o700)
        runner.chmod(0o500)
        if os.geteuid() == 0:
            os.chown(scratch, uid, gid)
            os.chown(runner, uid, gid)

        environment = {
            "HOME": str(scratch),
            "PATH": os.environ.get("PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TMPDIR": str(scratch),
        }
        try:
            completed = subprocess.run(
                [sys.executable, str(runner), str(source.resolve())],
                input=json.dumps(SCENARIOS),
                capture_output=True,
                text=True,
                cwd=scratch,
                env=environment,
                timeout=timeout_seconds,
                check=False,
                preexec_fn=_drop_privileges(uid, gid),
            )
        except subprocess.TimeoutExpired as error:
            raise RepairValidationError(
                f"submitted model repair test exceeded {timeout_seconds} seconds"
            ) from error

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise RepairValidationError(
            "submitted model could not run the repair regression test"
            + (f": {detail}" if detail else "")
        )

    payload_line = next(
        (
            line
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(RESULT_MARKER)
        ),
        None,
    )
    if payload_line is None:
        raise RepairValidationError("submitted model repair test returned no result")
    try:
        actual = json.loads(payload_line[len(RESULT_MARKER) :])
    except json.JSONDecodeError as error:
        raise RepairValidationError(
            "submitted model repair test returned invalid JSON"
        ) from error
    expected = [_expected_partition(scenario) for scenario in SCENARIOS]
    if actual != expected:
        raise RepairValidationError(
            "submitted model does not preserve the repaired chronological "
            "target-window behavior"
        )
