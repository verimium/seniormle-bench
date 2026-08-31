#!/usr/bin/env python3
"""Validate a top-level, data-free SeniorMLE-Bench distribution checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

GENERATED_SUFFIXES = {".arrow", ".csv", ".npy", ".npz", ".parquet"}
GENERATED_PRIVATE_FILES = {
    Path("tests/private/anti_cheat_report.json"),
    Path("tests/private/encounter_codes.npy"),
    Path("tests/private/reference_scores.json"),
    Path("tests/private/truth.json"),
}
RUNTIME_ROOTS = {".data", ".git", ".setup-work", ".venvs", "jobs", "tasks"}
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
MAX_SOURCE_FILE_BYTES = 1024 * 1024
TASK_ID_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def safe_relative_path(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe {label}: {value!r}")
    return path


def load_source_manifest(path: Path) -> dict[str, Any]:
    manifest = load_object(path, "source manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("source manifest schema_version must be 1")
    raw_task_ids = manifest.get("task_ids")
    if (
        not isinstance(raw_task_ids, list)
        or not raw_task_ids
        or not all(isinstance(task_id, str) for task_id in raw_task_ids)
        or not all(TASK_ID_PATTERN.fullmatch(task_id) for task_id in raw_task_ids)
        or raw_task_ids != sorted(set(raw_task_ids))
    ):
        raise TypeError("source manifest task_ids must be sorted and unique")
    if not isinstance(manifest.get("files"), dict):
        raise TypeError("source manifest files must be an object")
    return manifest


def is_generated_task_file(relative: Path) -> bool:
    if relative.name == ".gitkeep":
        return False
    data_roots = (Path("environment/public/data"), Path("tests/public/data"))
    if any(relative.is_relative_to(root) for root in data_roots):
        return relative.suffix.lower() in GENERATED_SUFFIXES
    return relative in GENERATED_PRIVATE_FILES


def is_generated_distribution_file(relative: Path, task_ids: set[str]) -> bool:
    return (
        len(relative.parts) > 1
        and relative.parts[0] in task_ids
        and is_generated_task_file(Path(*relative.parts[1:]))
    )


def source_files(root: Path, task_ids: set[str]) -> set[Path]:
    files: set[Path] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not relative.parts or relative.parts[0] in RUNTIME_ROOTS:
            continue
        if IGNORED_PARTS.intersection(relative.parts) or path.name == ".DS_Store":
            continue
        if is_generated_distribution_file(relative, task_ids):
            continue
        if path.is_file() or path.is_symlink():
            files.add(relative)
    return files


def validate_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    raw_files = manifest["files"]
    assert isinstance(raw_files, dict)
    expected: set[Path] = set()
    for relative_text, raw_entry in sorted(raw_files.items()):
        try:
            relative = safe_relative_path(relative_text, "manifest path")
        except ValueError as error:
            findings.append(str(error))
            continue
        expected.add(relative)
        if not isinstance(raw_entry, dict):
            findings.append(f"invalid manifest entry: {relative}")
            continue
        path = root / relative
        if path.is_symlink() or not path.is_file():
            findings.append(f"manifest file is missing or not regular: {relative}")
            continue
        expected_sha = raw_entry.get("sha256")
        if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
            findings.append(f"manifest hash mismatch: {relative}")
        expected_executable = raw_entry.get("executable")
        actual_executable = bool(path.stat().st_mode & stat.S_IXUSR)
        if not isinstance(expected_executable, bool):
            findings.append(f"invalid executable flag: {relative}")
        elif actual_executable != expected_executable:
            findings.append(f"executable mode mismatch: {relative}")

    task_ids = set(manifest["task_ids"])
    actual = source_files(root, task_ids)
    expected_with_manifest = expected | {Path("source-manifest.json")}
    for relative in sorted(expected_with_manifest - actual):
        findings.append(f"distribution source is missing: {relative}")
    for relative in sorted(actual - expected_with_manifest):
        findings.append(f"undeclared distribution source: {relative}")
    return findings


def validate_task_manifest(task_root: Path, task_id: str) -> list[str]:
    findings: list[str] = []
    try:
        manifest = load_object(
            task_root / "task-source-manifest.json", "task source manifest"
        )
    except (TypeError, ValueError) as error:
        return [f"{task_id}: {error}"]
    if manifest.get("schema_version") != 1:
        findings.append(f"{task_id}: task source manifest schema_version must be 1")
    if manifest.get("task_id") != task_id:
        findings.append(f"{task_id}: task source manifest task_id differs")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        findings.append(f"{task_id}: task source manifest files must be non-empty")
        return findings
    for relative_text, raw_entry in sorted(raw_files.items()):
        try:
            relative = safe_relative_path(relative_text, "task manifest path")
        except ValueError as error:
            findings.append(f"{task_id}: {error}")
            continue
        if not isinstance(raw_entry, dict):
            findings.append(f"{task_id}: invalid task manifest entry: {relative}")
            continue
        path = task_root / relative
        if path.is_symlink() or not path.is_file():
            findings.append(f"{task_id}: static task file is missing: {relative}")
            continue
        digest = raw_entry.get("sha256")
        if not isinstance(digest, str) or sha256_file(path) != digest:
            findings.append(f"{task_id}: static task hash mismatch: {relative}")
        executable = raw_entry.get("executable")
        actual_executable = bool(path.stat().st_mode & stat.S_IXUSR)
        if not isinstance(executable, bool) or executable != actual_executable:
            findings.append(f"{task_id}: static task mode mismatch: {relative}")
    return findings


def validate_task(root: Path, task_id: str) -> list[str]:
    task_root = root / task_id
    findings: list[str] = []
    required = (
        "task.toml",
        "instruction.md",
        "environment",
        "solution",
        "tests",
        "setup.sh",
        ".setup/config.json",
        ".setup/setup_task.py",
        "task-source-manifest.json",
    )
    if not task_root.is_dir():
        return [f"top-level task directory is missing: {task_id}"]
    for relative_text in required:
        path = task_root / relative_text
        if not path.exists() or path.is_symlink():
            findings.append(f"{task_id}: required task path is missing: {relative_text}")
    setup = task_root / "setup.sh"
    if setup.is_file() and setup.stat().st_mode & stat.S_IXUSR == 0:
        findings.append(f"{task_id}: setup.sh is not executable")

    task_toml = task_root / "task.toml"
    if task_toml.is_file():
        try:
            raw_task = tomllib.loads(task_toml.read_text(encoding="utf-8"))
            name = raw_task["task"]["name"]
            if not isinstance(name, str) or name.rsplit("/", 1)[-1] != task_id:
                findings.append(f"{task_id}: task.toml identity differs")
        except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
            findings.append(f"{task_id}: invalid task.toml: {error}")
    findings.extend(validate_task_manifest(task_root, task_id))
    return findings


def validate_top_level_layout(
    root: Path, task_ids: set[str]
) -> list[str]:
    findings: list[str] = []
    allowed_directories = {
        ".github",
        "scripts",
        *task_ids,
        *RUNTIME_ROOTS,
        *IGNORED_PARTS,
    }
    for path in root.iterdir():
        if path.is_dir() and path.name not in allowed_directories:
            findings.append(f"unexpected top-level directory: {path.name}")
    if (root / "rl-environment").exists():
        findings.append("private authoring tree must not exist at repository root")
    if (root / "setup.sh").exists():
        findings.append("setup.sh must be task-local, not repository-wide")
    return findings


def validate_source_sizes(root: Path, task_ids: set[str]) -> list[str]:
    findings: list[str] = []
    for relative in sorted(source_files(root, task_ids)):
        path = root / relative
        if path.is_symlink():
            findings.append(f"symlink is not allowed: {relative}")
        elif path.stat().st_size > MAX_SOURCE_FILE_BYTES:
            findings.append(f"source file exceeds 1 MiB: {relative}")
        elif path.suffix.lower() in GENERATED_SUFFIXES:
            findings.append(f"generated data exists outside a task data path: {relative}")
    return findings


def validate_tracked_data(root: Path, task_ids: set[str]) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        return []
    findings: list[str] = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        if is_generated_distribution_file(relative, task_ids) and (
            root / relative
        ).is_file():
            findings.append(f"generated task data is tracked by Git: {relative}")
    return findings


def validate_harbor_packages(root: Path, task_ids: list[str]) -> list[str]:
    findings: list[str] = []
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for task_id in task_ids:
        validator = (
            root
            / task_id
            / ".setup/source/rl-environment/scripts/validate_task_package.py"
        )
        if not validator.is_file():
            findings.append(f"{task_id}: canonical task validator is missing")
            continue
        completed = subprocess.run(
            [sys.executable, str(validator), str(root / task_id)],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            details = (completed.stdout or completed.stderr).strip()
            findings.append(f"{task_id}: Harbor package validation failed: {details}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    root = args.path.expanduser().resolve()
    try:
        manifest = load_source_manifest(root / "source-manifest.json")
    except (TypeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    task_ids = manifest["task_ids"]
    assert isinstance(task_ids, list)
    task_id_set = set(task_ids)
    findings = [
        *validate_manifest(root, manifest),
        *validate_top_level_layout(root, task_id_set),
        *validate_source_sizes(root, task_id_set),
        *validate_tracked_data(root, task_id_set),
    ]
    for task_id in task_ids:
        findings.extend(validate_task(root, task_id))
    findings.extend(validate_harbor_packages(root, task_ids))
    if findings:
        print("FAIL: invalid code-only task distribution", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(
        f"PASS: {len(task_ids)} top-level Harbor tasks; "
        f"{len(manifest['files'])} source files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
