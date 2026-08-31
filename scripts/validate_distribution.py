#!/usr/bin/env python3
"""Validate a code-only SeniorMLE-Bench distribution checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import sys
from pathlib import Path

GENERATED_ROOTS = {".data", ".git", ".venvs", "jobs", "tasks"}
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
FORBIDDEN_SUFFIXES = {".arrow", ".csv", ".npy", ".npz", ".parquet"}
FORBIDDEN_NAMES = {
    "anti_cheat_report.json",
    "encounter_codes.npy",
    "reference_scores.json",
    "truth.json",
}
MAX_SOURCE_FILE_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid source manifest: {error}") from error
    if payload.get("schema_version") != 1:
        raise ValueError("source manifest schema_version must be 1")
    if not isinstance(payload.get("files"), dict):
        raise TypeError("source manifest files must be an object")
    if not isinstance(payload.get("task_ids"), list) or not payload["task_ids"]:
        raise TypeError("source manifest task_ids must be a non-empty list")
    return payload


def validate_manifest(root: Path, manifest: dict[str, object]) -> list[str]:
    findings: list[str] = []
    raw_files = manifest["files"]
    assert isinstance(raw_files, dict)
    for relative_text, raw_entry in sorted(raw_files.items()):
        if not isinstance(relative_text, str) or not isinstance(raw_entry, dict):
            findings.append(f"invalid manifest entry: {relative_text!r}")
            continue
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            findings.append(f"unsafe manifest path: {relative_text}")
            continue
        path = root / relative
        if path.is_symlink() or not path.is_file():
            findings.append(f"manifest file is missing or not regular: {relative_text}")
            continue
        expected_sha = raw_entry.get("sha256")
        if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
            findings.append(f"manifest hash mismatch: {relative_text}")
        expected_executable = raw_entry.get("executable")
        actual_executable = bool(path.stat().st_mode & stat.S_IXUSR)
        if not isinstance(expected_executable, bool):
            findings.append(f"invalid executable flag: {relative_text}")
        elif actual_executable != expected_executable:
            findings.append(f"executable mode mismatch: {relative_text}")
    return findings


def validate_code_only_tree(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in GENERATED_ROOTS:
            continue
        if IGNORED_PARTS.intersection(relative.parts):
            continue
        if path.is_symlink():
            findings.append(f"symlink is not allowed: {relative}")
            continue
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"generated data file is tracked: {relative}")
        elif path.stat().st_size > MAX_SOURCE_FILE_BYTES:
            findings.append(f"source file exceeds 1 MiB: {relative}")
    setup = root / "setup.sh"
    if not setup.is_file() or setup.stat().st_mode & stat.S_IXUSR == 0:
        findings.append("setup.sh is missing or not executable")
    return findings


def validate_task_ids(root: Path, manifest: dict[str, object]) -> list[str]:
    scripts = root / "rl-environment/scripts"
    sys.path.insert(0, str(scripts))
    try:
        from harbor_task import HarborTaskError, discover_harbor_task_profiles
    except ImportError as error:
        return [f"could not import the task profile reader: {error}"]
    try:
        profiles = discover_harbor_task_profiles(root / "rl-environment/environments")
    except HarborTaskError as error:
        return [f"could not discover task profiles: {error}"]
    actual = [profile.task.task_id for profile in profiles]
    expected = manifest["task_ids"]
    if actual != expected:
        return [f"task IDs differ: expected {expected!r}, found {actual!r}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    root = args.path.expanduser().resolve()
    try:
        manifest = load_manifest(root / "source-manifest.json")
    except (TypeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    findings = [
        *validate_manifest(root, manifest),
        *validate_code_only_tree(root),
        *validate_task_ids(root, manifest),
    ]
    if findings:
        print("FAIL: invalid code-only task distribution", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1
    print(
        f"PASS: {len(manifest['task_ids'])} code-only tasks; "
        f"{len(manifest['files'])} source files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
