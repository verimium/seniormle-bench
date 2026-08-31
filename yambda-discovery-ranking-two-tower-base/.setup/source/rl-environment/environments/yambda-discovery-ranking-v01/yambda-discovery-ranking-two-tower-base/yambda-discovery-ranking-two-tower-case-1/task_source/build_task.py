"""Build the standalone Yambda two-tower improvement task."""

from __future__ import annotations

import sys
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent
TASK_SOURCES_DIR = PROFILE_DIR.parent.parent / "task_source"
if str(TASK_SOURCES_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_SOURCES_DIR))

from yambda_two_tower import builder as _builder

PROFILE = _builder.load_profile(PROFILE_DIR, variant="case-1")
TASK_CONTRACT = PROFILE.contract
TASK_ID = PROFILE.task_id
TASK_BUDGET_MINUTES = PROFILE.budget_minutes
REQUIREMENTS = _builder.REQUIREMENTS


def write_task_brief(out: Path) -> None:
    _builder.write_task_brief(PROFILE, out)


def write_task_runtime_contract(out: Path) -> None:
    _builder.write_task_runtime_contract(PROFILE, out)


def write_static_task_sources(out: Path) -> None:
    _builder.write_static_task_sources(PROFILE, out)


def verify_production_model_delta() -> None:
    _builder.verify_production_model_delta(PROFILE)


def load_recorded_case_model_scores() -> dict[str, dict[str, dict[str, object]]]:
    return _builder.load_recorded_case_model_scores(PROFILE)


def main() -> int:
    return _builder.main(PROFILE)


if __name__ == "__main__":
    raise SystemExit(main())
