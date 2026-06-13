#!/usr/bin/env python3
"""Read the Harbor task contract used by local benchmark tooling."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

TASK_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
PYTHON_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
SUPPORTED_SCHEMA_VERSION = "1.4"


class HarborTaskError(ValueError):
    """Raised when a checked-in Harbor task contract is malformed."""


@dataclass(frozen=True)
class HarborTask:
    schema_version: str
    task_id: str
    name: str
    version: str
    budget_seconds: int
    verifier_timeout_seconds: int
    python_version: str
    thresholds: dict[str, float]


def validate_task_id(value: Any, field: str = "task ID") -> str:
    if not isinstance(value, str) or TASK_ID_PATTERN.fullmatch(value) is None:
        raise HarborTaskError(
            f"{field} must contain only lowercase letters, numbers, and hyphens"
        )
    return value


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise HarborTaskError(f"{key} must be a TOML table")
    return value


def _positive_whole_seconds(value: Any, field: str) -> int:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
        or not float(value).is_integer()
    ):
        raise HarborTaskError(f"{field} must be a positive whole number of seconds")
    return int(value)


def parse_harbor_task(raw: Any, *, expected_task_id: str | None = None) -> HarborTask:
    if not isinstance(raw, dict):
        raise HarborTaskError("task.toml must contain a TOML table")
    schema_version = raw.get("schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise HarborTaskError(f"schema_version must be {SUPPORTED_SCHEMA_VERSION!r}")

    task = _table(raw, "task")
    name = task.get("name")
    if not isinstance(name, str) or name.count("/") != 1:
        raise HarborTaskError("task.name must use the Harbor org/name form")
    organization, task_id = name.split("/", 1)
    validate_task_id(organization, "task organization")
    validate_task_id(task_id)
    if expected_task_id is not None:
        validate_task_id(expected_task_id, "expected task ID")
        if task_id != expected_task_id:
            raise HarborTaskError(
                f"task.name ID {task_id!r} does not match {expected_task_id!r}"
            )
    version = task.get("version")
    if not isinstance(version, str) or not version.strip():
        raise HarborTaskError("task.version must be a non-empty string")

    agent = _table(raw, "agent")
    budget_seconds = _positive_whole_seconds(
        agent.get("timeout_sec"), "agent.timeout_sec"
    )
    verifier = _table(raw, "verifier")
    verifier_timeout = _positive_whole_seconds(
        verifier.get("timeout_sec"), "verifier.timeout_sec"
    )
    if verifier.get("environment_mode") != "separate":
        raise HarborTaskError(
            "verifier.environment_mode must be 'separate' for private verification"
        )
    verifier_environment = _table(verifier, "environment")
    if verifier_environment.get("network_mode") != "no-network":
        raise HarborTaskError("verifier.environment.network_mode must be 'no-network'")

    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, list) or "/app/solution" not in artifacts:
        raise HarborTaskError(
            "artifacts must include /app/solution for separate verification"
        )

    metadata = _table(raw, "metadata")
    runtime = _table(metadata, "runtime")
    python_version = runtime.get("python_version")
    if (
        not isinstance(python_version, str)
        or PYTHON_VERSION_PATTERN.fullmatch(python_version) is None
        or any(str(int(part)) != part for part in python_version.split("."))
    ):
        raise HarborTaskError(
            "metadata.runtime.python_version must be one X.Y.Z version"
        )

    verifier_metadata = _table(metadata, "verifier")
    raw_thresholds = _table(verifier_metadata, "thresholds")
    if not raw_thresholds:
        raise HarborTaskError("metadata.verifier.thresholds must not be empty")
    thresholds: dict[str, float] = {}
    for metric, value in raw_thresholds.items():
        if not isinstance(metric, str) or not metric:
            raise HarborTaskError("verifier threshold names must be non-empty")
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise HarborTaskError(
                f"metadata.verifier.thresholds.{metric} must be finite"
            )
        thresholds[metric] = float(value)

    return HarborTask(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        task_id=task_id,
        name=name,
        version=version,
        budget_seconds=budget_seconds,
        verifier_timeout_seconds=verifier_timeout,
        python_version=python_version,
        thresholds=thresholds,
    )


def read_harbor_task(path: Path, *, expected_task_id: str | None = None) -> HarborTask:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise HarborTaskError(f"failed to read {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise HarborTaskError(f"invalid TOML in {path}: {error}") from error
    return parse_harbor_task(raw, expected_task_id=expected_task_id)


def discover_harbor_tasks(environments_dir: Path) -> list[HarborTask]:
    """Return checked-in task profiles registered by a Harbor task.toml."""
    if not environments_dir.is_dir():
        raise HarborTaskError(
            f"environment profile directory does not exist: {environments_dir}"
        )
    tasks: list[HarborTask] = []
    for profile in sorted(environments_dir.iterdir()):
        if not profile.is_dir() or profile.name.startswith("."):
            continue
        task_path = profile / "task.toml"
        builder = profile / "build_task.py"
        if not task_path.is_file():
            raise HarborTaskError(f"task profile has no task.toml: {profile}")
        if not builder.is_file():
            raise HarborTaskError(f"task profile has no build_task.py: {profile}")
        tasks.append(read_harbor_task(task_path, expected_task_id=profile.name))
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise HarborTaskError("task profiles contain duplicate task IDs")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--field",
        choices=(
            "task_id",
            "name",
            "version",
            "budget_seconds",
            "verifier_timeout_seconds",
            "python_version",
            "thresholds",
        ),
        required=True,
    )
    parser.add_argument("--expect-task-id")
    args = parser.parse_args()
    try:
        task = read_harbor_task(args.path, expected_task_id=args.expect_task_id)
    except HarborTaskError as error:
        print(f"Invalid Harbor task: {error}", file=sys.stderr)
        return 2
    value = getattr(task, args.field)
    print(json.dumps(value, sort_keys=True) if isinstance(value, dict) else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
