#!/usr/bin/env python3
"""Validate a generated Harbor task package without reading hidden labels."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from harbor_task import HarborTask, HarborTaskError, read_harbor_task

EXACT_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^\s;]+$")
PYTHON_BASE = re.compile(
    r"^FROM\s+python:([^@\s]+)(?:@([^\s]+))?", re.MULTILINE | re.IGNORECASE
)
SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    path: str


class Validator:
    def __init__(self, task_dir: Path) -> None:
        self.task_dir = task_dir
        self.findings: list[Finding] = []

    def error(self, code: str, message: str, path: Path) -> None:
        self.findings.append(Finding("error", code, message, str(path)))

    def warning(self, code: str, message: str, path: Path) -> None:
        self.findings.append(Finding("warning", code, message, str(path)))

    def validate(self) -> list[Finding]:
        if not self.task_dir.is_dir():
            self.error(
                "task_directory_missing",
                "task directory does not exist",
                self.task_dir,
            )
            return self.findings

        environment = self.task_dir / "environment"
        tests = self.task_dir / "tests"
        solution = self.task_dir / "solution"
        for path, code in (
            (environment, "environment_directory_missing"),
            (tests, "tests_directory_missing"),
            (solution, "solution_directory_missing"),
        ):
            self._require_directory(path, code)

        self._forbid_legacy_files()
        task = self._validate_task_contract(self.task_dir / "task.toml")
        self._validate_instruction(self.task_dir / "instruction.md", task)
        self._validate_environment(environment, task)
        self._validate_tests(tests, task)
        self._validate_solution(solution)
        self._validate_shared_contract(environment, tests)
        self._validate_symlinks((environment, tests, solution))
        return self.findings

    def _require_directory(self, path: Path, code: str) -> None:
        if not path.is_dir():
            self.error(code, "required Harbor directory is missing", path)

    def _forbid_legacy_files(self) -> None:
        legacy = (
            "manifest.json",
            "task_manifest.json",
            "task_config.json",
            "public",
            "private",
            "judge",
            "solver-public",
        )
        for name in legacy:
            path = self.task_dir / name
            if path.exists() or path.is_symlink():
                self.error(
                    "legacy_package_artifact",
                    "generated packages must use the native Harbor layout",
                    path,
                )

    def _validate_task_contract(self, path: Path) -> HarborTask | None:
        if not path.is_file():
            self.error("task_toml_missing", "task.toml is required", path)
            return None
        try:
            return read_harbor_task(path, expected_task_id=self.task_dir.name)
        except HarborTaskError as error:
            self.error("task_toml_invalid", str(error), path)
            return None

    def _validate_instruction(self, path: Path, task: HarborTask | None) -> None:
        if not path.is_file():
            self.error("instruction_missing", "instruction.md is required", path)
            return
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            self.error("instruction_empty", "instruction.md must not be empty", path)
        if task is not None and task.budget_seconds % 60 == 0:
            expected = f"You have up to {task.budget_seconds // 60} minutes."
            if expected not in text:
                self.error(
                    "instruction_budget_mismatch",
                    f"instruction.md must state {expected!r}",
                    path,
                )

    def _validate_environment(self, environment: Path, task: HarborTask | None) -> None:
        if not environment.is_dir():
            return
        dockerfile = environment / "Dockerfile"
        self._validate_dockerfile(dockerfile, task, "agent")
        requirements = environment / "requirements.txt"
        self._validate_requirements(requirements)
        public = environment / "public"
        if not public.is_dir() or not any(path.is_file() for path in public.rglob("*")):
            self.error(
                "agent_inputs_missing",
                "environment/public must contain solver-visible task inputs",
                public,
            )
        if environment.is_dir():
            private_names = {
                "truth.json",
                "encounter_codes.npy",
                "anti_cheat_report.json",
                "evaluate_submission.py",
                "reference_solver.py",
                "baseline_solver.py",
                "production_solver.py",
                "model_metrics.json",
                "REFERENCE_SOLUTION.md",
                "test_recent_window_partition.py",
                "test_shared_negative_pool.py",
                "semantic_review.py",
            }
            private_directories = {"reference_solution"}
            if task is not None and task.review is not None:
                private_names.add(PurePosixPath(task.review.bug_summary).name)
            leaked = [
                path
                for path in environment.rglob("*")
                if path.is_file()
                and (
                    path.name in private_names
                    or bool(
                        private_directories.intersection(
                            path.relative_to(environment).parts
                        )
                    )
                )
            ]
            for path in leaked:
                self.error(
                    "private_asset_in_agent_image",
                    "private verifier asset leaked into the agent image context",
                    path,
                )

    def _validate_tests(self, tests: Path, task: HarborTask | None) -> None:
        if not tests.is_dir():
            return
        self._validate_dockerfile(tests / "Dockerfile", task, "verifier")
        self._validate_requirements(tests / "requirements.txt")
        test_script = tests / "test.sh"
        if not test_script.is_file():
            self.error("test_script_missing", "tests/test.sh is required", test_script)
        else:
            source = test_script.read_text(encoding="utf-8")
            if "/logs/verifier/reward.json" not in source and (
                "/logs/verifier/reward.txt" not in source
            ):
                self.error(
                    "reward_output_missing",
                    "tests/test.sh must produce a Harbor reward file",
                    test_script,
                )
            if test_script.stat().st_mode & 0o111 == 0:
                self.error(
                    "test_script_not_executable",
                    "tests/test.sh must be executable",
                    test_script,
                )
        private = tests / "private"
        if not private.is_dir() or not any(
            path.is_file() for path in private.rglob("*")
        ):
            self.error(
                "private_verifier_inputs_missing",
                "tests/private must contain evaluator-owned inputs",
                private,
            )
        if task is not None and task.review is not None:
            relative = PurePosixPath(task.review.bug_summary).relative_to("/tests")
            local_path = tests.joinpath(*relative.parts)
            if not local_path.is_file():
                self.error(
                    "semantic_review_input_missing",
                    "declared semantic-review input is missing",
                    local_path,
                )
            reviewer = tests / "semantic_review.py"
            if not reviewer.is_file():
                self.error(
                    "semantic_review_runner_missing",
                    "semantic-review task requires tests/semantic_review.py",
                    reviewer,
                )

    def _validate_solution(self, solution: Path) -> None:
        if not solution.is_dir():
            return
        solve_script = solution / "solve.sh"
        if not solve_script.is_file():
            self.error(
                "oracle_solution_missing",
                "solution/solve.sh is required for Harbor Oracle checks",
                solve_script,
            )
        elif solve_script.stat().st_mode & 0o111 == 0:
            self.error(
                "oracle_solution_not_executable",
                "solution/solve.sh must be executable",
                solve_script,
            )

    def _validate_dockerfile(
        self, path: Path, task: HarborTask | None, role: str
    ) -> None:
        if not path.is_file():
            self.error(
                f"{role}_dockerfile_missing",
                f"{role} Dockerfile is required",
                path,
            )
            return
        source = path.read_text(encoding="utf-8")
        match = PYTHON_BASE.search(source)
        if match is None:
            self.error(
                "python_base_missing",
                "Dockerfile must declare a tagged Python base image",
                path,
            )
        elif match.group(2) is None or SHA256_DIGEST.fullmatch(match.group(2)) is None:
            self.error(
                "python_base_digest_missing",
                "Dockerfile Python base image must be pinned by sha256 digest",
                path,
            )
        elif task is not None and not match.group(1).startswith(
            task.python_version + "-"
        ):
            self.error(
                "python_base_mismatch",
                "Dockerfile Python tag does not match task.toml runtime metadata",
                path,
            )

    def _validate_requirements(self, path: Path) -> None:
        if not path.is_file():
            self.error("requirements_missing", "exact requirements are required", path)
            return
        pins = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not pins:
            self.error(
                "requirements_empty", "requirements contain no distributions", path
            )
            return
        if any(EXACT_PIN.fullmatch(pin) is None for pin in pins):
            self.error(
                "requirements_not_exact",
                "requirements must contain only exact NAME==VERSION pins",
                path,
            )
        normalized = [
            re.sub(r"[-_.]+", "-", pin.split("==", 1)[0]).lower()
            for pin in pins
            if "==" in pin
        ]
        if len(normalized) != len(set(normalized)):
            self.error(
                "requirements_duplicate",
                "requirements contain a duplicate distribution",
                path,
            )

    def _validate_shared_contract(self, environment: Path, tests: Path) -> None:
        pairs = (
            (
                environment / "requirements.txt",
                tests / "requirements.txt",
                "requirements_mismatch",
            ),
            (
                self.task_dir / "task.toml",
                tests / "task.toml",
                "task_toml_mismatch",
            ),
            (
                environment / "public/ranking_evaluation.py",
                tests / "ranking_evaluation.py",
                "shared_evaluator_mismatch",
            ),
        )
        for public_path, verifier_path, code in pairs:
            if public_path.is_file() and verifier_path.is_file():
                if public_path.read_bytes() != verifier_path.read_bytes():
                    self.error(
                        code,
                        "agent and verifier contract copies must match byte for byte",
                        public_path,
                    )
            elif code != "shared_evaluator_mismatch":
                missing = public_path if not public_path.is_file() else verifier_path
                self.error(code, "required contract copy is missing", missing)

    def _validate_symlinks(self, directories: Iterable[Path]) -> None:
        for directory in directories:
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_symlink():
                    self.error(
                        "symlink_forbidden",
                        "Harbor task image contexts must not contain symlinks",
                        path,
                    )


def task_directories(root: Path, all_tasks: bool) -> list[Path]:
    if not all_tasks or not root.is_dir():
        return [root]
    return sorted(path for path in root.iterdir() if path.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Harbor task directory, or task-bank directory with --all",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="validate every direct child task directory",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable findings"
    )
    args = parser.parse_args()

    results = {
        str(path): Validator(path).validate()
        for path in task_directories(args.path, args.all)
    }
    errors = sum(
        finding.severity == "error"
        for findings in results.values()
        for finding in findings
    )
    warnings = sum(
        finding.severity == "warning"
        for findings in results.values()
        for finding in findings
    )
    if args.json:
        print(
            json.dumps(
                {
                    "errors": errors,
                    "warnings": warnings,
                    "tasks": {
                        path: [asdict(finding) for finding in findings]
                        for path, findings in results.items()
                    },
                },
                indent=2,
            )
        )
    else:
        for path, findings in results.items():
            if not findings:
                print(f"PASS {path}")
            for finding in findings:
                print(
                    f"{finding.severity.upper()} {path}: {finding.code}: "
                    f"{finding.message} ({finding.path})"
                )
        print(
            f"Summary: {errors} error(s), {warnings} warning(s), "
            f"{len(results)} task(s) checked."
        )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
