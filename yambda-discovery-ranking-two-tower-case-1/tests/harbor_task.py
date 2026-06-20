#!/usr/bin/env python3
"""Read the Harbor task contract used by local benchmark tooling."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import tomllib

TASK_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
PYTHON_VERSION_PATTERN = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
SUPPORTED_SCHEMA_VERSION = "1.4"


class HarborTaskError(ValueError):
    """Raised when a checked-in Harbor task contract is malformed."""


@dataclass(frozen=True)
class HarborReview:
    type: str
    artifact: str
    bug_summary: str
    stage_after: str
    gate: str
    threshold: float
    model: str
    reasoning_effort: str


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
    review: HarborReview | None

    @property
    def review_type(self) -> str | None:
        return None if self.review is None else self.review.type


@dataclass(frozen=True)
class HarborTaskProfile:
    directory: Path
    task: HarborTask


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


def _absolute_path_below(value: Any, root: str, field: str) -> str:
    if not isinstance(value, str):
        raise HarborTaskError(f"{field} must be an absolute container path")
    path = PurePosixPath(value)
    root_path = PurePosixPath(root)
    if (
        not path.is_absolute()
        or not path.is_relative_to(root_path)
        or path == root_path
    ):
        raise HarborTaskError(f"{field} must be below {root}")
    if ".." in path.parts:
        raise HarborTaskError(f"{field} must not contain '..'")
    return value


def _semantic_review(
    metadata: dict[str, Any], thresholds: dict[str, float]
) -> HarborReview | None:
    raw = metadata.get("review")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise HarborTaskError("metadata.review must be a TOML table")
    review_type = raw.get("type")
    if review_type != "llm-semantic-alignment":
        raise HarborTaskError("metadata.review.type must be 'llm-semantic-alignment'")
    stage_after = raw.get("stage_after")
    if not isinstance(stage_after, str) or stage_after not in thresholds:
        raise HarborTaskError(
            "metadata.review.stage_after must name a verifier threshold"
        )
    gate = raw.get("gate")
    if not isinstance(gate, str) or not gate or gate in thresholds:
        raise HarborTaskError(
            "metadata.review.gate must be a non-empty name distinct from metric gates"
        )
    threshold = raw.get("threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or not 0.0 < float(threshold) <= 1.0
    ):
        raise HarborTaskError("metadata.review.threshold must be in (0, 1]")
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HarborTaskError("metadata.review.model must be a non-empty string")
    reasoning_effort = raw.get("reasoning_effort")
    if reasoning_effort not in {"low", "medium", "high"}:
        raise HarborTaskError(
            "metadata.review.reasoning_effort must be low, medium, or high"
        )
    return HarborReview(
        type=review_type,
        artifact=_absolute_path_below(
            raw.get("artifact"), "/app/solution", "metadata.review.artifact"
        ),
        bug_summary=_absolute_path_below(
            raw.get("bug_summary"),
            "/tests/private",
            "metadata.review.bug_summary",
        ),
        stage_after=stage_after,
        gate=gate,
        threshold=float(threshold),
        model=model,
        reasoning_effort=reasoning_effort,
    )


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

    review = _semantic_review(metadata, thresholds)
    verifier_network_mode = verifier_environment.get("network_mode")
    if review is None:
        if verifier_network_mode != "no-network":
            raise HarborTaskError(
                "verifier.environment.network_mode must be 'no-network'"
            )
    else:
        if verifier_network_mode != "allowlist":
            raise HarborTaskError(
                "semantic-review verifier.environment.network_mode must be 'allowlist'"
            )
        allowed_hosts = verifier_environment.get("allowed_hosts")
        if (
            not isinstance(allowed_hosts, list)
            or not allowed_hosts
            or any(
                not isinstance(host, str) or not host.strip() for host in allowed_hosts
            )
        ):
            raise HarborTaskError(
                "semantic-review verifier.environment.allowed_hosts must not be empty"
            )

    return HarborTask(
        schema_version=SUPPORTED_SCHEMA_VERSION,
        task_id=task_id,
        name=name,
        version=version,
        budget_seconds=budget_seconds,
        verifier_timeout_seconds=verifier_timeout,
        python_version=python_version,
        thresholds=thresholds,
        review=review,
    )


def read_harbor_task(path: Path, *, expected_task_id: str | None = None) -> HarborTask:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise HarborTaskError(f"failed to read {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise HarborTaskError(f"invalid TOML in {path}: {error}") from error
    return parse_harbor_task(raw, expected_task_id=expected_task_id)


def discover_harbor_task_profiles(
    environments_dir: Path,
) -> list[HarborTaskProfile]:
    """Return task profiles from the nested environment hierarchy."""
    if not environments_dir.is_dir():
        raise HarborTaskError(
            f"environment profile directory does not exist: {environments_dir}"
        )
    profiles: list[HarborTaskProfile] = []
    for builder in sorted(environments_dir.rglob("build_task.py")):
        relative = builder.relative_to(environments_dir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        profile = builder.parent
        task_paths = [
            path
            for path in (profile / "task.toml", profile / "harbor/task.toml")
            if path.is_file()
        ]
        if not task_paths:
            raise HarborTaskError(f"task profile has no task.toml: {profile}")
        if len(task_paths) > 1:
            raise HarborTaskError(
                f"task profile has multiple task.toml contracts: {profile}"
            )
        expected_task_id = (
            profile.parent.name if profile.name == "task_source" else profile.name
        )
        task = read_harbor_task(task_paths[0], expected_task_id=expected_task_id)
        profiles.append(HarborTaskProfile(directory=profile, task=task))
    profiles.sort(key=lambda profile: profile.task.task_id)
    task_ids = [profile.task.task_id for profile in profiles]
    if len(task_ids) != len(set(task_ids)):
        raise HarborTaskError("task profiles contain duplicate task IDs")
    return profiles


def discover_harbor_tasks(environments_dir: Path) -> list[HarborTask]:
    """Return checked-in tasks registered anywhere in the hierarchy."""
    return [profile.task for profile in discover_harbor_task_profiles(environments_dir)]


def find_harbor_task_profile(environments_dir: Path, task_id: str) -> Path:
    """Resolve one task ID to its checked-in authoring profile."""
    task_id = validate_task_id(task_id)
    matches = [
        profile.directory
        for profile in discover_harbor_task_profiles(environments_dir)
        if profile.task.task_id == task_id
    ]
    if not matches:
        raise HarborTaskError(f"unknown task ID: {task_id}")
    if len(matches) != 1:
        raise HarborTaskError(f"task ID has multiple profiles: {task_id}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--field",
        choices=(
            "task_id",
            "name",
            "version",
            "budget_seconds",
            "verifier_timeout_seconds",
            "python_version",
            "thresholds",
            "review_type",
        ),
    )
    action.add_argument("--find-profile", metavar="TASK_ID")
    parser.add_argument("--expect-task-id")
    args = parser.parse_args()
    try:
        if args.find_profile is not None:
            print(find_harbor_task_profile(args.path, args.find_profile).resolve())
            return 0
        task = read_harbor_task(args.path, expected_task_id=args.expect_task_id)
    except HarborTaskError as error:
        print(f"Invalid Harbor task: {error}", file=sys.stderr)
        return 2
    if args.field is None:
        parser.error("--field is required unless --find-profile is used")
    value = getattr(task, args.field)
    print(json.dumps(value, sort_keys=True) if isinstance(value, dict) else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
