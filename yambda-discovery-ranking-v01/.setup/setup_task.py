#!/usr/bin/env python3
"""Hydrate one data-free SeniorMLE-Bench task in place."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

GENERATED_SUFFIXES = {".arrow", ".csv", ".npy", ".npz", ".parquet"}
GENERATED_PRIVATE_FILES = {
    Path("tests/private/anti_cheat_report.json"),
    Path("tests/private/encounter_codes.npy"),
    Path("tests/private/reference_scores.json"),
    Path("tests/private/truth.json"),
}
PROTECTED_BUILDER_OPTIONS = {
    "--code-only",
    "--env-dir",
    "--out",
    "--skip-env-sync",
    "--source-dir",
    "--use-current-python",
}


class SetupError(RuntimeError):
    """Raised when a task cannot be hydrated without violating its manifest."""


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
        raise SetupError(f"invalid {label} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise SetupError(f"{label} must be a JSON object: {path}")
    return value


def safe_relative_path(value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise SetupError(f"{label} must be a string")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SetupError(f"unsafe {label}: {value!r}")
    return path


def load_config(task_root: Path) -> tuple[str, Path, Path]:
    config = load_object(task_root / ".setup/config.json", "setup config")
    if config.get("schema_version") != 1:
        raise SetupError("setup config schema_version must be 1")
    task_id = config.get("task_id")
    if not isinstance(task_id, str) or task_id != task_root.name:
        raise SetupError(
            f"setup config task_id must match task directory {task_root.name!r}"
        )
    builder_relative = safe_relative_path(config.get("builder"), "builder path")
    source_root = task_root / ".setup/source"
    builder = source_root / builder_relative
    if builder.is_symlink() or not builder.is_file():
        raise SetupError(f"configured builder is missing: {builder}")
    return task_id, source_root, builder


def load_static_manifest(task_root: Path, task_id: str) -> dict[Path, dict[str, Any]]:
    path = task_root / "task-source-manifest.json"
    manifest = load_object(path, "task source manifest")
    if manifest.get("schema_version") != 1:
        raise SetupError("task source manifest schema_version must be 1")
    if manifest.get("task_id") != task_id:
        raise SetupError("task source manifest task_id does not match setup config")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise SetupError("task source manifest files must be a non-empty object")

    entries: dict[Path, dict[str, Any]] = {}
    for relative_text, raw_entry in raw_files.items():
        relative = safe_relative_path(relative_text, "manifest path")
        if not isinstance(raw_entry, dict):
            raise SetupError(f"invalid manifest entry for {relative}")
        digest = raw_entry.get("sha256")
        executable = raw_entry.get("executable")
        if not isinstance(digest, str) or len(digest) != 64:
            raise SetupError(f"invalid manifest hash for {relative}")
        if not isinstance(executable, bool):
            raise SetupError(f"invalid executable flag for {relative}")
        entries[relative] = raw_entry
    return entries


def verify_static_files(
    root: Path, entries: dict[Path, dict[str, Any]], label: str
) -> None:
    for relative, entry in sorted(entries.items()):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise SetupError(f"{label} static file is missing: {relative}")
        if sha256_file(path) != entry["sha256"]:
            raise SetupError(f"{label} static file differs: {relative}")
        executable = bool(path.stat().st_mode & stat.S_IXUSR)
        if executable != entry["executable"]:
            raise SetupError(f"{label} executable mode differs: {relative}")


def is_generated_file(relative: Path) -> bool:
    data_roots = (Path("environment/public/data"), Path("tests/public/data"))
    if any(relative.is_relative_to(root) for root in data_roots):
        return relative.suffix.lower() in GENERATED_SUFFIXES
    return relative in GENERATED_PRIVATE_FILES


def generated_files(root: Path) -> list[Path]:
    generated: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not is_generated_file(relative):
            continue
        if path.is_symlink() or not path.is_file():
            raise SetupError(f"generated task path must be a regular file: {relative}")
        generated.append(relative)
    return generated


def verify_built_task(
    built_task: Path, entries: dict[Path, dict[str, Any]]
) -> list[Path]:
    verify_static_files(built_task, entries, "built task")
    static_paths = set(entries)
    generated: list[Path] = []
    for path in sorted(built_task.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(built_task)
        if path.is_symlink() or not path.is_file():
            raise SetupError(f"built task contains a non-regular path: {relative}")
        if relative in static_paths:
            continue
        if not is_generated_file(relative):
            raise SetupError(f"builder produced an undeclared task file: {relative}")
        generated.append(relative)
    if not generated:
        raise SetupError("builder produced no generated task data")
    return generated


def run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    try:
        subprocess.run(command, cwd=cwd, env=environment, check=True)
    except FileNotFoundError as error:
        raise SetupError(f"required command not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        raise SetupError(
            f"command failed with exit code {error.returncode}: {shlex.join(command)}"
        ) from error


def reject_protected_arguments(arguments: list[str]) -> None:
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in PROTECTED_BUILDER_OPTIONS:
            raise SetupError(f"setup owns the {option} builder option")


def print_help(task_id: str) -> None:
    print(
        f"""Usage: ./{task_id}/setup.sh [builder arguments]

Downloads the pinned source dataset, verifies its checksums, and hydrates the
Harbor task in place. Common optional arguments are --force-download,
--force-env, --skip-download, and --skip-validate-source.

Environment variables:
  PYTHON                Python used to launch the builder (default: python3)
  SENIORMLE_DATA_DIR    Shared source-data cache
  SENIORMLE_ENVS_DIR    Shared task build-environment directory"""
    )


def install_generated_files(
    task_root: Path,
    built_task: Path,
    generated: list[Path],
    backup_root: Path,
    validator: Path,
    python: str,
    environment: dict[str, str],
) -> None:
    existing = generated_files(task_root)
    installed: list[Path] = []
    backed_up: list[Path] = []
    try:
        for relative in existing:
            source = task_root / relative
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            source.replace(backup)
            backed_up.append(relative)
        for relative in generated:
            source = built_task / relative
            target = task_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            installed.append(relative)
        run(
            [python, str(validator), str(task_root)],
            cwd=task_root,
            environment=environment,
        )
    except BaseException:
        for relative in reversed(installed):
            target = task_root / relative
            if target.is_file() or target.is_symlink():
                target.unlink()
        for relative in backed_up:
            backup = backup_root / relative
            target = task_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            backup.replace(target)
        raise


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    task_root = Path(__file__).resolve().parents[1]
    task_id, source_root, builder = load_config(task_root)
    if any(argument in {"-h", "--help"} for argument in arguments):
        print_help(task_id)
        return 0
    reject_protected_arguments(arguments)

    entries = load_static_manifest(task_root, task_id)
    verify_static_files(task_root, entries, "distributed task")

    repository_root = task_root.parent
    data_dir = Path(
        os.environ.get(
            "SENIORMLE_DATA_DIR", str(repository_root / ".data/yambda/50m")
        )
    ).expanduser().resolve()
    envs_dir = Path(
        os.environ.get("SENIORMLE_ENVS_DIR", str(repository_root / ".venvs"))
    ).expanduser().resolve()
    work_root = repository_root / ".setup-work"
    work_root.mkdir(parents=True, exist_ok=True)
    python = os.environ.get("PYTHON", "python3")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    validator = source_root / "rl-environment/scripts/validate_task_package.py"
    if validator.is_symlink() or not validator.is_file():
        raise SetupError(f"task package validator is missing: {validator}")

    with tempfile.TemporaryDirectory(
        prefix=f".{task_id}.setup-", dir=work_root
    ) as directory:
        temporary_root = Path(directory)
        built_task = temporary_root / task_id
        command = [
            python,
            str(builder),
            *arguments,
            "--source-dir",
            str(data_dir),
            "--env-dir",
            str(envs_dir / task_id),
            "--out",
            str(built_task),
        ]
        run(command, cwd=source_root, environment=environment)
        generated = verify_built_task(built_task, entries)
        install_generated_files(
            task_root,
            built_task,
            generated,
            temporary_root / "previous-data",
            validator,
            python,
            environment,
        )

    print(f"\nReady Harbor task: {task_root}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SetupError as error:
        print(f"setup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
