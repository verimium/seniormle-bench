#!/usr/bin/env python3
"""Validate a task outcome and adapt it to Harbor's numeric reward contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from harbor_task import HarborTaskError, validate_task_id

BASE_KEYS = {
    "schema_version",
    "task_id",
    "status",
    "reward",
    "passed",
    "metrics",
    "thresholds",
    "gates",
}


class VerifierResultError(ValueError):
    """Raised when verifier output is not a canonical episode outcome."""


def _number(value: Any, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
    ):
        raise VerifierResultError(f"{field} must be a finite number")
    return float(value)


def _numeric_mapping(value: Any, field: str) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise VerifierResultError(f"{field} must be an object")
    parsed: dict[str, int | float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise VerifierResultError(f"{field} keys must be non-empty strings")
        _number(item, f"{field}.{key}")
        parsed[key] = item
    return parsed


def validate_verifier_result(
    raw: Any, *, expected_task_id: str | None = None
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise VerifierResultError("verifier result must be a JSON object")
    status = raw.get("status")
    expected_keys = BASE_KEYS | ({"reason"} if status == "invalid" else set())
    if set(raw) != expected_keys:
        raise VerifierResultError(
            f"{status or 'unknown'} result keys must be exactly {sorted(expected_keys)}"
        )
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise VerifierResultError("schema_version must be integer 1")
    try:
        task_id = validate_task_id(raw["task_id"])
        if expected_task_id is not None:
            validate_task_id(expected_task_id, "expected task ID")
    except HarborTaskError as error:
        raise VerifierResultError(str(error)) from error
    if expected_task_id is not None and task_id != expected_task_id:
        raise VerifierResultError(
            f"result task_id {task_id!r} does not match {expected_task_id!r}"
        )
    if status not in {"scored", "invalid"}:
        raise VerifierResultError("status must be 'scored' or 'invalid'")
    if not isinstance(raw["passed"], bool):
        raise VerifierResultError("passed must be a boolean")

    reward = _number(raw["reward"], "reward")
    if not 0.0 <= reward <= 1.0:
        raise VerifierResultError("reward must be between 0 and 1")
    metrics = _numeric_mapping(raw["metrics"], "metrics")
    thresholds = _numeric_mapping(raw["thresholds"], "thresholds")
    gates = raw["gates"]
    if not isinstance(gates, dict) or any(
        not isinstance(key, str) or not key or not isinstance(value, bool)
        for key, value in gates.items()
    ):
        raise VerifierResultError("gates must map non-empty string names to booleans")
    if set(gates) != set(thresholds):
        raise VerifierResultError("gates and thresholds must have identical keys")

    if status == "invalid":
        reason = raw["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise VerifierResultError("invalid results require a non-empty reason")
        if reward != 0.0 or raw["passed"] or metrics or any(gates.values()):
            raise VerifierResultError(
                "invalid results must have zero reward, passed=false, no metrics, "
                "and all gates false"
            )
    else:
        if not metrics:
            raise VerifierResultError("scored results require at least one metric")
        missing_metrics = set(thresholds) - set(metrics)
        if missing_metrics:
            raise VerifierResultError(
                "threshold metrics missing from metrics: "
                + ", ".join(sorted(missing_metrics))
            )
        inconsistent_gates = [
            name
            for name, threshold in thresholds.items()
            if gates[name] != (_number(metrics[name], f"metrics.{name}") >= threshold)
        ]
        if inconsistent_gates:
            raise VerifierResultError(
                "gates disagree with metric thresholds: "
                + ", ".join(sorted(inconsistent_gates))
            )
        if raw["passed"] != all(gates.values()):
            raise VerifierResultError("passed must equal the conjunction of gates")
    return raw


def read_verifier_result(
    path: Path, *, expected_task_id: str | None = None
) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise VerifierResultError(f"failed to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise VerifierResultError(f"invalid JSON in {path}: {error}") from error
    return validate_verifier_result(raw, expected_task_id=expected_task_id)


def result_passed(
    raw: Any,
    *,
    expected_task_id: str | None = None,
    allow_legacy: bool = False,
) -> bool:
    """Return a verdict, optionally accepting pre-contract calibration records."""
    try:
        return bool(
            validate_verifier_result(raw, expected_task_id=expected_task_id)["passed"]
        )
    except VerifierResultError:
        if (
            allow_legacy
            and isinstance(raw, dict)
            and isinstance(raw.get("passed"), bool)
            and raw.get("status") in {"scored", "invalid", "pass"}
        ):
            if (
                expected_task_id is not None
                and "task_id" in raw
                and raw["task_id"] != expected_task_id
            ):
                raise
            return bool(raw["passed"])
        raise


def harbor_rewards(
    raw: Any, *, expected_task_id: str | None = None
) -> dict[str, int | float]:
    """Return the numeric reward mapping consumed by Harbor.

    The scalar ``reward`` remains the RL signal. ``passed`` is numeric because
    Harbor reward files only accept numbers. Task metrics are propagated for
    analysis, while thresholds, gates, and invalid-artifact reasons remain in
    the adjacent detailed ``result.json`` verifier log.
    """
    result = validate_verifier_result(raw, expected_task_id=expected_task_id)
    reserved = {"reward", "passed"}
    collisions = reserved & set(result["metrics"])
    if collisions:
        raise VerifierResultError(
            "metrics use Harbor-reserved reward keys: " + ", ".join(sorted(collisions))
        )
    return {
        "reward": float(result["reward"]),
        "passed": int(result["passed"]),
        **result["metrics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--expect-task-id")
    parser.add_argument(
        "--harbor-reward-output",
        type=Path,
        help="write Harbor reward.json after validating the detailed result",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        result = read_verifier_result(args.path, expected_task_id=args.expect_task_id)
        if args.harbor_reward_output is not None:
            rewards = harbor_rewards(result, expected_task_id=args.expect_task_id)
            args.harbor_reward_output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.harbor_reward_output.with_suffix(
                args.harbor_reward_output.suffix + ".tmp"
            )
            temporary.write_text(json.dumps(rewards, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.harbor_reward_output)
    except VerifierResultError as error:
        print(f"Invalid verifier result: {error}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(f"PASS {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
