"""Build standalone Harbor tasks from the shared Yambda two-tower family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FAMILY_DIR = Path(__file__).resolve().parent
ASSETS_DIR = FAMILY_DIR / "assets"


def find_repo_root(start: Path) -> Path:
    for candidate in start.parents:
        if (candidate / "rl-environment/scripts").is_dir():
            return candidate
    raise RuntimeError(f"could not locate repository root from {start}")


REPO_ROOT = find_repo_root(FAMILY_DIR)
SCRIPTS_DIR = REPO_ROOT / "rl-environment/scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from harbor_task import HarborTask, HarborTaskError, read_harbor_task
from verifier_result import VerifierResultError, validate_verifier_result

DEFAULT_SOURCE_DIR = REPO_ROOT / ".context/datasets/yambda/50m"
REQUIREMENTS = ASSETS_DIR / "requirements.txt"

Variant = Literal["base", "case-1"]


@dataclass(frozen=True)
class TaskProfile:
    """Variant-specific inputs used to materialize one standalone task."""

    directory: Path
    entrypoint: Path
    variant: Variant
    contract: HarborTask

    @property
    def task_id(self) -> str:
        return self.contract.task_id

    @property
    def task_toml_path(self) -> Path:
        return self.harbor_dir / "task.toml"

    @property
    def harbor_dir(self) -> Path:
        """Return the checked-in source for the generated Harbor package."""
        return self.directory / "harbor"

    @property
    def private_dir(self) -> Path:
        """Return evaluator-only authoring sources for this profile."""
        return self.directory / "private"

    @property
    def default_out(self) -> Path:
        return REPO_ROOT / ".context/harbor/tasks" / self.task_id

    @property
    def default_env_dir(self) -> Path:
        return REPO_ROOT / ".context/senior-mle-bench-envs" / self.task_id

    @property
    def required_python_version(self) -> tuple[int, ...]:
        return tuple(int(part) for part in self.contract.python_version.split("."))

    @property
    def budget_minutes(self) -> int:
        seconds = self.contract.budget_seconds
        if seconds % 60:
            raise SystemExit("task agent timeout must be a whole number of minutes")
        return seconds // 60

    @property
    def is_case(self) -> bool:
        return self.variant == "case-1"


def load_profile(directory: Path, *, variant: Variant) -> TaskProfile:
    """Load and validate a thin task overlay."""
    directory = directory.resolve()
    expected_task_id = {
        "base": "yambda-discovery-ranking-two-tower-base",
        "case-1": "yambda-discovery-ranking-two-tower-case-1",
    }[variant]
    profile_task_id = (
        directory.parent.name if directory.name == "task_source" else directory.name
    )
    if profile_task_id != expected_task_id:
        raise SystemExit(f"{variant} profile node must be named {expected_task_id!r}")
    task_toml = directory / "harbor/task.toml"
    try:
        contract = read_harbor_task(task_toml, expected_task_id=expected_task_id)
    except HarborTaskError as error:
        raise SystemExit(f"invalid Harbor task contract: {error}") from error
    if contract.review is not None:
        raise SystemExit("two-tower task contracts must not configure semantic review")
    if contract.budget_seconds % 60:
        raise SystemExit("task agent timeout must be a whole number of minutes")
    profile = TaskProfile(
        directory=directory,
        entrypoint=directory / "build_task.py",
        variant=variant,
        contract=contract,
    )
    return profile


DATASET_REVISION = "dd6f3a19eef5866e346c3270e098baa641a44948"
DATASET_REPO = (
    f"https://huggingface.co/datasets/yandex/yambda/resolve/{DATASET_REVISION}"
)
DATASET_SUBDIR = "flat/50m"

DAY = 86_400
GAP = 1_800
LAST_TIMESTAMP = 26_000_000
WINDOW_DAYS = 7
WINDOW = WINDOW_DAYS * DAY
TEST_START = LAST_TIMESTAMP - WINDOW
VAL_END = TEST_START - GAP
VAL_START = VAL_END - WINDOW
TRAIN_END = VAL_START - GAP

LISTEN_POSITIVE = 50
LISTEN_STRONG = 80
TARGET_RECENT_DAYS = 90
TARGET_MIN_POSITIVES = 20
CANDIDATE_SIZE = 30_000
TOP_N = 100
REFERENCE_RECENT_DAYS = 21
REFERENCE_K = 120
REFERENCE_HISTORY = 80

EVENT_NAMES = ("listens", "likes", "dislikes", "unlikes", "undislikes")
EXPLICIT_NAMES = ("likes", "dislikes", "unlikes", "undislikes")

REQUIRED_COLUMNS = {
    "listens": {
        "uid",
        "timestamp",
        "item_id",
        "is_organic",
        "played_ratio_pct",
        "track_length_seconds",
    },
    "likes": {"uid", "timestamp", "item_id", "is_organic"},
    "dislikes": {"uid", "timestamp", "item_id", "is_organic"},
    "unlikes": {"uid", "timestamp", "item_id", "is_organic"},
    "undislikes": {"uid", "timestamp", "item_id", "is_organic"},
}

SOURCE_SHA256 = {
    "dislikes": "64f240a6d62b8acdfa6efae660a54d10ebc64f9f2f0aeeef8e6bb0dbe7dc2802",
    "likes": "694087077dbacfcc1d22a5ca85cc6bd8ab182361933406e60500624c6a422bc4",
    "listens": "eed9cbd094af1e189507d2f8132a0dc9653b90e65480125c7cdccd601e0592d1",
    "undislikes": "8cdbcf6e7e79491d560637057ba8ac078dd1610a3fab15442b6f12e45069ef85",
    "unlikes": "532213992934de765874691ac145324078e61987df26c88e1d671d23f8853180",
}

Ranker = Callable[[int], list[int]]


def log(message: str) -> None:
    print(message, flush=True)


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def require_python_version(profile: TaskProfile) -> None:
    current = sys.version_info[:3]
    if current != profile.required_python_version:
        current_text = ".".join(str(part) for part in current)
        raise SystemExit(
            f"this task must be built with CPython {profile.contract.python_version}; "
            f"current interpreter is {current_text}"
        )


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True, cwd=cwd)
    except FileNotFoundError as error:
        raise SystemExit(f"required command not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"command failed with exit code {error.returncode}: " + " ".join(command)
        ) from error


def sync_environment(profile: TaskProfile, env_dir: Path, force: bool = False) -> Path:
    if shutil.which("uv") is None:
        raise SystemExit("uv is required to build this task")

    python = env_dir / "bin/python"
    if force and env_dir.exists():
        shutil.rmtree(env_dir)
    if not python.exists():
        version = profile.contract.python_version
        run(["uv", "python", "install", version])
        run(["uv", "venv", "--python", version, str(env_dir)])
    run(
        [
            "uv",
            "pip",
            "sync",
            "--python",
            str(python),
            "--torch-backend",
            "cpu",
            str(REQUIREMENTS),
        ]
    )
    return python


def rerun_in_environment(
    profile: TaskProfile, python: Path, env_dir: Path, argv: list[str]
) -> None:
    command = [
        str(python),
        str(profile.entrypoint),
        "--use-current-python",
        "--env-dir",
        str(env_dir),
        *argv,
    ]
    os.execv(str(python), command)


def source_url(name: str) -> str:
    return f"{DATASET_REPO}/{DATASET_SUBDIR}/{name}.parquet?download=true"


def download_file(
    url: str, destination: Path, expected_sha256: str, force: bool = False
) -> None:
    if destination.is_file() and not force:
        log(f"exists: {destination}")
        verify_checksum(destination, expected_sha256)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    log(f"download: {url}")
    log(f"     to: {destination}")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    except (OSError, urllib.error.URLError) as error:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"failed to download {url}: {error}") from error
    partial.replace(destination)
    verify_checksum(destination, expected_sha256)


def validate_source(path: Path, required_columns: set[str]) -> None:
    import pyarrow.parquet as pq

    try:
        schema = pq.ParquetFile(path).schema_arrow
    except Exception as error:
        raise SystemExit(
            f"failed to read parquet schema for {path}: {error}"
        ) from error
    missing = sorted(required_columns - set(schema.names))
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(missing)}")
    log(f"valid: {path}")


def prepare_source_data(source_dir: Path, force: bool, validate: bool) -> None:
    for name, columns in REQUIRED_COLUMNS.items():
        path = source_dir / f"{name}.parquet"
        download_file(source_url(name), path, SOURCE_SHA256[name], force=force)
        if validate:
            validate_source(path, columns)


def load_build_dependencies() -> None:
    """Import heavy builder dependencies only for full package builds."""
    global np, pd, sparse

    try:
        import numpy as np  # type: ignore[import-not-found]
        import pandas as pd  # type: ignore[import-not-found]
        from scipy import sparse  # type: ignore[import-not-found]
    except ImportError as error:
        raise SystemExit(
            "numpy, pandas, and scipy are required to build this task; "
            "run build_task.py without --skip-env-sync"
        ) from error


def _replace_exactly_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise SystemExit(f"case model no longer contains the expected {label} delta")
    return source.replace(old, new, 1)


def _sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _model_overlay(profile: TaskProfile) -> dict[str, object]:
    path = profile.directory / "model_overlay.json"
    if not path.is_file():
        raise SystemExit(f"case model overlay not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid case model overlay: {error}") from error
    if payload.get("schema_version") != 2:
        raise SystemExit("case model overlay must use schema_version 2")
    replacements = payload.get("replacements")
    if not isinstance(replacements, list) or not replacements:
        raise SystemExit("case model overlay requires a non-empty replacements list")
    return payload


def reference_model_path(profile: TaskProfile) -> Path:
    """Return the profile-owned reference solver source."""
    if profile.is_case:
        return profile.private_dir / "reference_solution/reference_solver.py"
    return ASSETS_DIR / "reference_solver.py"


def render_reference_model(profile: TaskProfile) -> str:
    """Return the reference implementation selected for this profile."""
    path = reference_model_path(profile)
    if not path.is_file():
        raise SystemExit(f"reference model source not found: {path}")
    return path.read_text(encoding="utf-8")


def render_production_model(profile: TaskProfile) -> str:
    """Apply the case overlay to the reference with exact-match guards."""
    if not profile.is_case:
        raise SystemExit("only case profiles materialize a deployed current model")
    payload = _model_overlay(profile)
    source = render_reference_model(profile)
    for replacement in payload["replacements"]:
        if not isinstance(replacement, dict):
            raise SystemExit("case model replacements must be JSON objects")
        try:
            label = str(replacement["label"])
            reference = str(replacement["reference"])
            production = str(replacement["production"])
        except KeyError as error:
            raise SystemExit(
                f"case model replacement is missing {error.args[0]}"
            ) from error
        source = _replace_exactly_once(source, reference, production, label)
    return source


def verify_production_model_delta(profile: TaskProfile) -> None:
    """Prove the generated public model is the intended production baseline."""
    if not profile.is_case:
        return
    payload = _model_overlay(profile)
    reference = render_reference_model(profile)
    production = render_production_model(profile)
    if _sha256_text(reference) != payload.get("reference_sha256"):
        raise SystemExit("case reference model hash does not match the case overlay")
    if _sha256_text(production) != payload.get("production_sha256"):
        raise SystemExit(
            "generated current model hash does not match the production fixture"
        )


def write_private_reference_solver(profile: TaskProfile, out: Path) -> None:
    """Install the profile-owned reference and evaluator-only calibration sources."""
    reference_source = render_reference_model(profile)
    model_source = ASSETS_DIR / "two_tower_model.py"
    if not model_source.is_file():
        raise SystemExit(f"private model source not found: {model_source}")

    for directory in (out / "solution", out / "tests/private"):
        (directory / "reference_solver.py").write_text(
            reference_source, encoding="utf-8"
        )
        shutil.copyfile(model_source, directory / "two_tower_model.py")
    if not profile.is_case:
        return

    (out / "tests/private/production_solver.py").write_text(
        render_production_model(profile), encoding="utf-8"
    )
    reference_dir = out / "tests/private/reference_solution"
    reference_dir.mkdir()
    (reference_dir / "reference_solver.py").write_text(
        reference_source, encoding="utf-8"
    )
    shutil.copyfile(model_source, reference_dir / "two_tower_model.py")


def write_public_current_model(profile: TaskProfile, out: Path) -> None:
    """Install the deployed production model used as the case baseline."""
    if not profile.is_case:
        return
    current_model = out / "environment/public/variant/current_model"
    current_model.mkdir()
    (current_model / "train_current_model.py").write_text(
        render_production_model(profile), encoding="utf-8"
    )
    shutil.copyfile(
        ASSETS_DIR / "two_tower_model.py", current_model / "two_tower_model.py"
    )


def write_private_readme(profile: TaskProfile, out: Path) -> None:
    """Install the task-author diagnosis and build-time regression artifact."""
    if not profile.is_case:
        return
    for name in (
        "README.md",
        "model_metrics.json",
        "test_shared_negative_pool.py",
    ):
        shutil.copyfile(profile.private_dir / name, out / "tests/private" / name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise SystemExit(
            f"{path} sha256 mismatch:\n"
            f"  expected: {expected_sha256}\n"
            f"    actual: {actual}"
        )


def write_anti_cheat_report(out: Path, report: dict[str, object]) -> None:
    (out / "tests/private/anti_cheat_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def write_task_runtime_contract(profile: TaskProfile, out: Path) -> None:
    """Install the Harbor contract and both image-context templates."""
    if not REQUIREMENTS.is_file():
        raise SystemExit(f"missing task requirements: {REQUIREMENTS}")
    shutil.copyfile(profile.task_toml_path, out / "task.toml")
    shutil.copyfile(profile.task_toml_path, out / "tests/task.toml")
    for directory in ("environment", "tests"):
        shutil.copyfile(REQUIREMENTS, out / directory / "requirements.txt")
    templates = {
        "environment/Dockerfile": ASSETS_DIR / "environment.Dockerfile",
        "tests/Dockerfile": profile.harbor_dir / "tests/Dockerfile",
        "tests/test.sh": profile.harbor_dir / "tests/test.sh",
        "solution/solve.sh": profile.harbor_dir / "solution/solve.sh",
    }
    for relative, source in templates.items():
        if not source.is_file():
            raise SystemExit(f"Harbor task template not found: {source}")
        destination = out / relative
        shutil.copyfile(source, destination)
        if destination.name in {"test.sh", "solve.sh"}:
            destination.chmod(0o755)
    for helper in ("harbor_task.py", "verifier_result.py"):
        shutil.copyfile(SCRIPTS_DIR / helper, out / "tests" / helper)


def require_sources(source_dir: Path) -> None:
    missing = [
        source_dir / f"{name}.parquet"
        for name in EVENT_NAMES
        if not (source_dir / f"{name}.parquet").is_file()
    ]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"missing Yambda source files:\n{formatted}")
    for name in EVENT_NAMES:
        verify_checksum(source_dir / f"{name}.parquet", SOURCE_SHA256[name])


def load_sources(source_dir: Path) -> dict[str, pd.DataFrame]:
    log("1/11 loading Yambda source tables")
    frames: dict[str, pd.DataFrame] = {}
    frames["listens"] = pd.read_parquet(
        source_dir / "listens.parquet",
        columns=[
            "uid",
            "timestamp",
            "item_id",
            "is_organic",
            "played_ratio_pct",
            "track_length_seconds",
        ],
    )
    for name in EXPLICIT_NAMES:
        frames[name] = pd.read_parquet(
            source_dir / f"{name}.parquet",
            columns=["uid", "timestamp", "item_id", "is_organic"],
        )
    return frames


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame.timestamp < TRAIN_END].copy()
    val = frame[(frame.timestamp >= VAL_START) & (frame.timestamp < VAL_END)].copy()
    test = frame[
        (frame.timestamp >= TEST_START) & (frame.timestamp < LAST_TIMESTAMP)
    ].copy()
    return train, val, test


def build_catalog_and_targets(
    train_listens: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, set[int], set[int]]:
    positive = train_listens[train_listens.played_ratio_pct >= LISTEN_POSITIVE]

    item_counts = (
        positive.groupby("item_id", sort=False)
        .agg(
            positive_listens=("item_id", "size"),
            organic_positive_listens=("is_organic", "sum"),
        )
        .reset_index()
        .sort_values(["positive_listens", "item_id"], ascending=[False, True])
        .head(CANDIDATE_SIZE)
        .reset_index(drop=True)
    )
    item_counts["popularity_rank"] = np.arange(1, len(item_counts) + 1, dtype=np.int32)

    recent = positive[positive.timestamp >= TRAIN_END - TARGET_RECENT_DAYS * DAY]
    recent_counts = (
        recent.groupby("uid", sort=False).size().rename("recent_positive_listens")
    )
    history_counts = (
        positive.groupby("uid", sort=False).size().rename("history_positive_listens")
    )
    targets = pd.concat([recent_counts, history_counts], axis=1).fillna(0).reset_index()
    targets = targets[targets.recent_positive_listens >= TARGET_MIN_POSITIVES].copy()
    targets = targets.sort_values("uid").reset_index(drop=True)

    catalog = set(map(int, item_counts.item_id))
    target_ids = set(map(int, targets.uid))
    return item_counts, targets, catalog, target_ids


def write_public_data(
    out: Path,
    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    catalog: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    log("2/11 writing public train/validation data")
    data_dir = out / "environment/public/data"
    for name, (train, val, _test) in splits.items():
        train.to_parquet(
            data_dir / f"train_{name}.parquet", index=False, compression="zstd"
        )
        val.to_parquet(
            data_dir / f"val_{name}.parquet", index=False, compression="zstd"
        )
    catalog.to_parquet(
        data_dir / "candidate_catalog.parquet", index=False, compression="zstd"
    )
    targets.to_parquet(
        data_dir / "target_users.parquet", index=False, compression="zstd"
    )


def write_upstream_sources(out: Path) -> None:
    """Copy the checked-in eligibility builder into the temporary package."""
    source = ASSETS_DIR / "build_eligibility.py"
    if not source.is_file():
        raise SystemExit(f"eligibility builder source not found: {source}")
    shutil.copyfile(source, out / "environment/public/upstream/build_eligibility.py")


def run_public_eligibility_build(out: Path) -> None:
    log("3/11 building cutoff-specific candidate-set data")
    script = (out / "environment/public/upstream/build_eligibility.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=script.parent,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"public eligibility build failed:\n{result.stdout}\n{result.stderr}"
        )
    log("  " + result.stdout.strip())


def verify_upstream_eligibility(
    out: Path,
    stage: str,
    seen: dict[int, set[int]],
    catalog: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    """Prove one generated candidate artifact exactly matches its history."""
    expected_uids = np.sort(targets.uid.to_numpy(dtype=np.int64, copy=False))
    expected_items = catalog.sort_values("popularity_rank").item_id.to_numpy(
        dtype=np.int64, copy=False
    )
    artifact_path = out / f"environment/public/upstream/{stage}_eligible_candidates.npz"
    with np.load(artifact_path) as artifact:
        uids = artifact["uids"].astype(np.int64, copy=False)
        item_ids = artifact["item_ids"].astype(np.int64, copy=False)
        packed = artifact["packed_eligible"]
        if not np.array_equal(uids, expected_uids):
            raise SystemExit("upstream eligibility artifact has incorrect target users")
        if not np.array_equal(item_ids, expected_items):
            raise SystemExit(
                "upstream eligibility artifact has incorrect catalog ordering"
            )
        item_column = {int(item): column for column, item in enumerate(item_ids)}
        for row, raw_uid in enumerate(uids):
            uid = int(raw_uid)
            known = seen.get(uid, set())
            eligible = np.unpackbits(
                packed[row], count=len(item_ids), bitorder="little"
            )
            if int(eligible.sum()) != len(item_ids) - len(known):
                raise SystemExit(f"{stage} eligibility count mismatch for uid {uid}")
            if any(eligible[item_column[item]] for item in known):
                raise SystemExit(
                    f"{stage} eligibility leaked an encounter for uid {uid}"
                )


def finalise_public_candidate_inputs(out: Path) -> None:
    """Expose only candidate-set data; remove the internal build implementation."""
    upstream = out / "environment/public/upstream"
    data = out / "environment/public/data"
    shutil.move(
        upstream / "validation_eligible_candidates.npz",
        data / "validation_eligible_candidates.npz",
    )
    shutil.move(
        upstream / "submission_eligible_candidates.npz",
        data / "test_eligible_candidates.npz",
    )
    shutil.rmtree(upstream)


def update_seen(
    seen: dict[int, set[int]],
    frame: pd.DataFrame,
    target_ids: set[int],
    catalog: set[int],
) -> None:
    subset = frame[frame.uid.isin(target_ids) & frame.item_id.isin(catalog)]
    for uid, values in subset.groupby("uid", sort=False).item_id:
        seen[int(uid)].update(map(int, values.unique()))


def build_seen(
    history_frames: dict[str, pd.DataFrame], target_ids: set[int], catalog: set[int]
) -> dict[int, set[int]]:
    seen: dict[int, set[int]] = defaultdict(set)
    for name in EVENT_NAMES:
        update_seen(seen, history_frames[name], target_ids, catalog)
    return dict(seen)


def pack_pairs(rows: pd.DataFrame) -> np.ndarray:
    uid = rows.uid.to_numpy(dtype=np.uint64, copy=False)
    item = rows.item_id.to_numpy(dtype=np.uint64, copy=False)
    return (uid << np.uint64(32)) | item


def flatten_seen_codes(seen: dict[int, set[int]]) -> np.ndarray:
    chunks = []
    for uid, items in seen.items():
        values = np.fromiter(items, dtype=np.uint64, count=len(items))
        chunks.append((np.uint64(uid) << np.uint64(32)) | values)
    if not chunks:
        return np.empty(0, dtype=np.uint64)
    codes = np.concatenate(chunks)
    codes.sort()
    return codes


def active_channel(
    positive: pd.DataFrame,
    clear: pd.DataFrame,
    target_ids: set[int],
    catalog: set[int],
) -> pd.DataFrame:
    """Return active pairs; reversal wins when timestamps tie."""
    on = positive[positive.uid.isin(target_ids) & positive.item_id.isin(catalog)][
        ["uid", "timestamp", "item_id"]
    ].copy()
    off = clear[clear.uid.isin(target_ids) & clear.item_id.isin(catalog)][
        ["uid", "timestamp", "item_id"]
    ].copy()
    on["active"] = 1
    on["tie_priority"] = 0
    off["active"] = 0
    off["tie_priority"] = 1
    events = pd.concat([on, off], ignore_index=True)
    latest = events.sort_values(
        ["uid", "item_id", "timestamp", "tie_priority"], kind="mergesort"
    ).drop_duplicates(["uid", "item_id"], keep="last")
    return latest[latest.active == 1][["uid", "item_id"]].copy()


def build_expected_state(
    train_frames: dict[str, pd.DataFrame], target_ids: set[int], catalog: set[int]
) -> tuple[pd.DataFrame, dict[int, set[int]], dict[int, set[int]]]:
    active_likes = active_channel(
        train_frames["likes"], train_frames["unlikes"], target_ids, catalog
    )
    active_dislikes = active_channel(
        train_frames["dislikes"], train_frames["undislikes"], target_ids, catalog
    )
    state = pd.concat(
        [
            active_likes.assign(state="liked"),
            active_dislikes.assign(state="disliked"),
        ],
        ignore_index=True,
    )
    state = state.sort_values(["uid", "item_id", "state"]).reset_index(drop=True)
    liked = {
        int(uid): set(map(int, values))
        for uid, values in active_likes.groupby("uid", sort=False).item_id
    }
    disliked = {
        int(uid): set(map(int, values))
        for uid, values in active_dislikes.groupby("uid", sort=False).item_id
    }
    return state, liked, disliked


def build_truth(
    listens: pd.DataFrame,
    likes: pd.DataFrame,
    seen: dict[int, set[int]],
    target_ids: set[int],
    catalog: set[int],
) -> dict[int, dict[int, int]]:
    strong = listens[
        listens.uid.isin(target_ids)
        & listens.item_id.isin(catalog)
        & (listens.is_organic == 1)
        & (listens.played_ratio_pct >= LISTEN_STRONG)
    ]
    organic_likes = likes[
        likes.uid.isin(target_ids)
        & likes.item_id.isin(catalog)
        & (likes.is_organic == 1)
    ]

    graded: dict[int, dict[int, int]] = defaultdict(dict)
    for uid, values in strong.groupby("uid", sort=False).item_id:
        known = seen.get(int(uid), set())
        for raw_item in values.unique():
            item = int(raw_item)
            if item not in known:
                graded[int(uid)][item] = 1
    for uid, values in organic_likes.groupby("uid", sort=False).item_id:
        known = seen.get(int(uid), set())
        for raw_item in values.unique():
            item = int(raw_item)
            if item not in known:
                graded[int(uid)][item] = 3
    return dict(graded)


def serialise_truth(graded: dict[int, dict[int, int]]) -> dict[str, dict[str, object]]:
    return {
        "graded": {
            str(uid): [[item, grade] for item, grade in sorted(gold.items())]
            for uid, gold in sorted(graded.items())
        },
    }


def verify_temporal_isolation(
    out: Path,
    validation_seen: dict[int, set[int]],
    submission_seen: dict[int, set[int]],
    validation_graded: dict[int, dict[int, int]],
    test_graded: dict[int, dict[int, int]],
) -> dict[str, object]:
    """Fail the build if public labels can be replayed as hidden candidates."""
    leaked_validation_relevance = 0
    for uid, gold in validation_graded.items():
        leaked_validation_relevance += len(
            set(gold).difference(submission_seen.get(int(uid), set()))
        )
    if leaked_validation_relevance:
        raise SystemExit(
            "submission eligibility admits public-validation relevance labels"
        )

    ineligible_test_relevance = 0
    for uid, gold in test_graded.items():
        ineligible_test_relevance += len(
            set(gold).intersection(submission_seen.get(int(uid), set()))
        )
    if ineligible_test_relevance:
        raise SystemExit("hidden truth contains an item ineligible at test cutoff")

    newly_excluded = sum(
        len(items.difference(validation_seen.get(uid, set())))
        for uid, items in submission_seen.items()
    )
    if newly_excluded == 0:
        raise SystemExit("validation history did not change submission eligibility")

    evaluator_owned_names = {
        *(f"test_{name}.parquet" for name in EVENT_NAMES),
        "truth.json",
        "encounter_codes.npy",
        "reference_solver.py",
        "production_solver.py",
        "model_metrics.json",
        "README.md",
        "test_shared_negative_pool.py",
        "semantic_review.py",
    }
    forbidden_public_files = [
        path
        for path in (out / "environment/public").rglob("*")
        if path.is_file() and path.name in evaluator_owned_names
    ]
    if forbidden_public_files:
        raise SystemExit(
            "evaluator-owned artifacts leaked into public/: "
            + ", ".join(map(str, forbidden_public_files))
        )

    report: dict[str, object] = {
        "validation_cutoff_encounter_pairs": sum(map(len, validation_seen.values())),
        "submission_cutoff_encounter_pairs": sum(map(len, submission_seen.values())),
        "new_pairs_excluded_after_validation": newly_excluded,
        "validation_relevant_pairs": sum(map(len, validation_graded.values())),
        "validation_relevant_pairs_eligible_at_submission": leaked_validation_relevance,
        "hidden_relevant_pairs": sum(map(len, test_graded.values())),
        "hidden_relevant_pairs_ineligible_at_submission": ineligible_test_relevance,
        "public_contains_hidden_labels": False,
    }
    return report


def ndcg_at_k(ranked: list[int], gold: dict[int, int], k: int) -> float:
    gains = np.asarray([gold.get(item, 0) for item in ranked[:k]], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum((np.power(2.0, gains) - 1.0) * discounts))
    ideal = np.asarray(sorted(gold.values(), reverse=True)[:k], dtype=np.float64)
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float(np.sum((np.power(2.0, ideal) - 1.0) * ideal_discounts))
    return dcg / idcg if idcg else 0.0


def score_ranker(
    ranker: Ranker,
    graded: dict[int, dict[int, int]],
) -> dict[str, float | int]:
    ranked = {uid: ranker(uid)[:TOP_N] for uid in graded}
    quality = float(
        np.mean([ndcg_at_k(ranked[uid], gold, 10) for uid, gold in graded.items()])
    )
    return {
        "quality_ndcg@10": round(quality, 6),
        "quality_users": len(graded),
    }


def fit_similarity(
    positive_events: pd.DataFrame,
    candidate_items: np.ndarray,
    recent_days: int,
    k: int,
    history_end: int,
    organic_weight: float = 1.0,
    like_events: pd.DataFrame | None = None,
    like_weight: float = 0.0,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    cutoff = history_end - recent_days * DAY
    recent = positive_events[
        (positive_events.timestamp >= cutoff)
        & positive_events.item_id.isin(candidate_items)
    ].copy()
    items = np.asarray(candidate_items, dtype=np.int64)
    users = np.sort(recent.uid.unique())
    item_index = {int(value): index for index, value in enumerate(items)}
    user_index = {int(value): index for index, value in enumerate(users)}
    rows = recent.uid.map(user_index).to_numpy()
    cols = recent.item_id.map(item_index).to_numpy()
    values = np.where(recent.is_organic.to_numpy() == 1, organic_weight, 1.0).astype(
        np.float32
    )

    if like_events is not None and like_weight:
        relevant = like_events[
            (like_events.timestamp >= cutoff)
            & like_events.uid.isin(users)
            & like_events.item_id.isin(items)
        ]
        rows = np.concatenate([rows, relevant.uid.map(user_index).to_numpy()])
        cols = np.concatenate([cols, relevant.item_id.map(item_index).to_numpy()])
        values = np.concatenate(
            [values, np.full(len(relevant), like_weight, dtype=np.float32)]
        )

    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))
    matrix.sum_duplicates()
    matrix.data = np.log1p(matrix.data)
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=0)).A1
    norms[norms == 0] = 1.0
    normalised = matrix.multiply(sparse.csr_matrix(1.0 / norms)).tocsr()
    similarity = (normalised.T @ normalised).tocsr()

    data_out: list[np.ndarray] = []
    indices_out: list[np.ndarray] = []
    indptr = [0]
    for row_id in range(similarity.shape[0]):
        start, end = similarity.indptr[row_id], similarity.indptr[row_id + 1]
        data = similarity.data[start:end]
        indices = similarity.indices[start:end]
        if len(data) > k:
            selected = np.argpartition(-data, k)[:k]
            data, indices = data[selected], indices[selected]
        data_out.append(data)
        indices_out.append(indices)
        indptr.append(indptr[-1] + len(data))
    return (
        sparse.csr_matrix(
            (
                np.concatenate(data_out),
                np.concatenate(indices_out),
                np.asarray(indptr),
            ),
            shape=similarity.shape,
        ),
        items,
    )


def build_profile(
    positive: pd.DataFrame,
    target_ids: set[int],
    catalog_items: np.ndarray,
    history_end: int,
    history_depth: int = REFERENCE_HISTORY,
) -> dict[int, list[tuple[int, float]]]:
    history = (
        positive[positive.uid.isin(target_ids) & positive.item_id.isin(catalog_items)]
        .groupby(["uid", "item_id"], sort=False)
        .agg(
            plays=("item_id", "size"),
            last=("timestamp", "max"),
            organic=("is_organic", "mean"),
        )
        .reset_index()
    )
    history["weight"] = (
        (1.0 + np.log1p(history.plays))
        * (1.0 + history.organic)
        * np.exp(-(history_end - history["last"]) / (120 * DAY))
    )
    ordered = history.sort_values(["uid", "weight"], ascending=[True, False])
    return {
        int(uid): list(zip(group.item_id.astype(int), group.weight.astype(float)))[
            :history_depth
        ]
        for uid, group in ordered.groupby("uid", sort=False)
    }


def make_filtered_ranker(
    similarity: sparse.csr_matrix,
    item_ids: np.ndarray,
    profile: dict[int, list[tuple[int, float]]],
    seen: dict[int, set[int]],
    fallback: np.ndarray,
) -> Ranker:
    index_of = {int(value): index for index, value in enumerate(item_ids)}

    def rank(uid: int) -> list[int]:
        known = seen.get(int(uid), set())
        scores: dict[int, float] = {}
        for item, weight in profile.get(int(uid), []):
            column = index_of.get(int(item))
            if column is None:
                continue
            start, end = similarity.indptr[column], similarity.indptr[column + 1]
            for neighbour, value in zip(
                similarity.indices[start:end], similarity.data[start:end]
            ):
                candidate = int(item_ids[neighbour])
                if candidate not in known:
                    scores[candidate] = scores.get(candidate, 0.0) + weight * float(
                        value
                    )
        ranked = [item for item, _ in sorted(scores.items(), key=lambda pair: -pair[1])]
        selected = set(ranked)
        for raw_item in fallback:
            if len(ranked) >= TOP_N:
                break
            item = int(raw_item)
            if item not in known and item not in selected:
                ranked.append(item)
                selected.add(item)
        return ranked[:TOP_N]

    return rank


def write_public_evaluator(out: Path) -> None:
    """Install the public evaluator and its shared evaluation library."""
    for name in ("evaluate_public.py", "ranking_evaluation.py"):
        source = ASSETS_DIR / name
        if not source.is_file():
            raise SystemExit(f"public evaluation source not found: {source}")
        shutil.copyfile(source, out / "environment/public" / name)


def run_public_evaluator_smoke_check(out: Path) -> None:
    """Exercise the public evaluator with an eligibility-safe popularity ranking."""
    log("6/11 smoke-checking the public discovery evaluator")
    public = out / "environment/public"
    with np.load(public / "data/validation_eligible_candidates.npz") as artifact:
        uids = artifact["uids"].astype(np.int64, copy=False)
        item_ids = artifact["item_ids"].astype(np.int64, copy=False)
        packed = artifact["packed_eligible"]
        rows: list[tuple[int, int, int]] = []
        for row, uid in enumerate(uids):
            eligible = np.unpackbits(
                packed[row], count=len(item_ids), bitorder="little"
            ).astype(bool, copy=False)
            selected = item_ids[eligible][:TOP_N]
            rows.extend(
                (int(uid), int(item), rank)
                for rank, item in enumerate(selected, start=1)
            )
    scratch = out / "tests/private/public_evaluator_smoke.parquet"
    pd.DataFrame(rows, columns=["uid", "item_id", "rank"]).to_parquet(
        scratch, index=False
    )
    script = (public / "evaluate_public.py").resolve()
    result = subprocess.run(
        [sys.executable, str(script), str(scratch.resolve())],
        capture_output=True,
        text=True,
        cwd=public,
        check=False,
    )
    scratch.unlink(missing_ok=True)
    if result.returncode != 0:
        raise SystemExit(
            f"public discovery evaluation failed:\n{result.stdout}\n{result.stderr}"
        )
    try:
        report = json.loads(result.stdout)
        scores = report["scores"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise SystemExit(f"public discovery evaluator returned invalid JSON: {error}")
    log(f"  evaluator smoke ranking: quality_ndcg@10={scores['quality_ndcg@10']:.6f}")


def write_private_evaluator(profile: TaskProfile, out: Path) -> None:
    """Install the evaluator-only entrypoint and its shared evaluation library.

    The verifier gets its own copy of ``ranking_evaluation.py`` so it never
    imports code from the agent image. Both copies come from the same source.
    """
    sources = {
        "evaluate_submission.py": profile.harbor_dir / "tests/evaluate_submission.py",
        "ranking_evaluation.py": ASSETS_DIR / "ranking_evaluation.py",
    }
    for name, source in sources.items():
        if not source.is_file():
            raise SystemExit(f"private evaluation source not found: {source}")
        shutil.copyfile(source, out / "tests" / name)


def write_verifier_public_inputs(out: Path) -> None:
    """Copy only public files needed by the isolated verifier image."""
    source = out / "environment/public/data"
    destination = out / "tests/public/data"
    for name in ("candidate_catalog.parquet", "target_users.parquet"):
        shutil.copyfile(source / name, destination / name)


def run_adversarial_judge_checks(
    profile: TaskProfile,
    out: Path,
    targets: pd.DataFrame,
    catalog: pd.DataFrame,
    submission_seen: dict[int, set[int]],
    validation_graded: dict[int, dict[int, int]],
    report: dict[str, object],
) -> None:
    """Exercise the real judge against raw and eligibility-filtered label replay."""
    scratch = out / "tests/private/.anti_cheat_attempts"
    raw_dir = scratch / "raw_replay"
    filtered_dir = scratch / "filtered_replay"
    raw_dir.mkdir(parents=True)
    filtered_dir.mkdir(parents=True)
    catalog_items = list(map(int, catalog.sort_values("popularity_rank").item_id))

    def rows_for(filtered: bool) -> list[tuple[int, int, int]]:
        rows: list[tuple[int, int, int]] = []
        for raw_uid in targets.uid:
            uid = int(raw_uid)
            known = submission_seen.get(uid, set()) if filtered else set()
            gold = validation_graded.get(uid, {})
            ranked = [
                item
                for item, _grade in sorted(
                    gold.items(), key=lambda pair: (-pair[1], pair[0])
                )
                if item not in known
            ][:TOP_N]
            selected = set(ranked)
            for item in catalog_items:
                if len(ranked) >= TOP_N:
                    break
                if item not in known and item not in selected:
                    ranked.append(item)
                    selected.add(item)
            if len(ranked) != TOP_N:
                raise SystemExit(f"anti-cheat fixture incomplete for uid {uid}")
            rows.extend((uid, item, rank) for rank, item in enumerate(ranked, 1))
        return rows

    columns = ["uid", "item_id", "rank"]
    pd.DataFrame(rows_for(False), columns=columns).to_parquet(
        raw_dir / "submission.parquet", index=False
    )
    pd.DataFrame(rows_for(True), columns=columns).to_parquet(
        filtered_dir / "submission.parquet", index=False
    )
    judge = (out / "tests/evaluate_submission.py").resolve()

    try:
        raw_result = subprocess.run(
            [sys.executable, str(judge), str(raw_dir.resolve())],
            capture_output=True,
            text=True,
            cwd=judge.parent,
            check=False,
        )
        try:
            raw_payload = json.loads(raw_result.stdout)
            validate_verifier_result(raw_payload, expected_task_id=profile.task_id)
        except (json.JSONDecodeError, VerifierResultError) as error:
            raise SystemExit(f"raw replay judge output is invalid: {error}")
        if raw_result.returncode != 0 or raw_payload.get("status") != "invalid":
            raise SystemExit("the private judge accepted raw validation-label replay")
        if raw_payload.get("reward") != 0 or raw_payload.get("passed") is not False:
            raise SystemExit("invalid replay did not produce a zero-reward outcome")

        filtered_result = subprocess.run(
            [sys.executable, str(judge), str(filtered_dir.resolve())],
            capture_output=True,
            text=True,
            cwd=judge.parent,
            check=False,
        )
        try:
            filtered_payload = json.loads(filtered_result.stdout)
            validate_verifier_result(filtered_payload, expected_task_id=profile.task_id)
        except (json.JSONDecodeError, VerifierResultError) as error:
            raise SystemExit(f"filtered replay judge output is invalid: {error}")
        if (
            filtered_result.returncode != 0
            or filtered_payload.get("status") != "scored"
        ):
            raise SystemExit("the private judge could not score filtered replay")
        if filtered_payload.get("passed"):
            raise SystemExit(
                "eligibility-filtered validation-label replay passes the gates"
            )

        report.update(
            {
                "raw_validation_label_replay_rejected_by_judge": True,
                "filtered_validation_label_replay_passed": False,
                "filtered_validation_label_replay_metrics": filtered_payload["metrics"],
            }
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def write_task_brief(profile: TaskProfile, out: Path) -> None:
    source = profile.directory / "instruction.md"
    if not source.is_file():
        raise SystemExit(f"task brief source not found: {source}")
    source_text = source.read_text(encoding="utf-8")
    expected_budget_text = f"You have up to {profile.budget_minutes} minutes."
    if expected_budget_text not in source_text:
        raise SystemExit(
            "instruction.md must state the task.toml time budget exactly as "
            f"{expected_budget_text!r}"
        )
    threshold = profile.contract.thresholds["quality_ndcg@10"]
    threshold_text = f"{threshold:g}"
    if threshold_text in source_text:
        raise SystemExit(
            "agent-facing instruction.md must not disclose the private quality "
            f"threshold {threshold_text!r}"
        )
    shutil.copyfile(source, out / "instruction.md")


def gates_pass(scores: dict[str, float | int], thresholds: dict[str, float]) -> bool:
    return all(
        float(scores[name]) >= threshold for name, threshold in thresholds.items()
    )


def load_recorded_case_model_scores(
    profile: TaskProfile,
) -> dict[str, dict[str, dict[str, object]]]:
    """Load the authoritative Harbor calibration without refitting either model."""
    if not profile.is_case:
        raise SystemExit("recorded model calibration is only defined for case-1")
    path = profile.private_dir / "model_metrics.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid recorded model metrics: {error}") from error
    if payload.get("schema_version") != 1:
        raise SystemExit("recorded model metrics must use schema_version 1")
    environment = payload.get("environment")
    models = payload.get("models")
    if not isinstance(environment, str) or not environment:
        raise SystemExit("recorded model metrics require a calibration environment")
    if not isinstance(models, dict):
        raise SystemExit("recorded model metrics require a models object")

    overlay = _model_overlay(profile)
    expected_models = {
        "production_two_tower": overlay.get("production_sha256"),
        "shared_negative_reference_two_tower": overlay.get("reference_sha256"),
    }
    scores: dict[str, dict[str, dict[str, object]]] = {
        "public_validation": {},
        "hidden_test": {},
    }
    for model_name, expected_hash in expected_models.items():
        model = models.get(model_name)
        if not isinstance(model, dict):
            raise SystemExit(f"recorded metrics are missing {model_name}")
        source_hash = model.get("source_sha256")
        if source_hash != expected_hash:
            raise SystemExit(f"recorded {model_name} source hash is stale")
        for stage in scores:
            stage_metrics = model.get(stage)
            if not isinstance(stage_metrics, dict):
                raise SystemExit(f"recorded {model_name}/{stage} metrics are missing")
            quality = stage_metrics.get("quality_ndcg@10")
            users = stage_metrics.get("quality_users")
            if not isinstance(quality, (int, float)) or not 0.0 <= quality <= 1.0:
                raise SystemExit(f"recorded {model_name}/{stage} NDCG is invalid")
            if not isinstance(users, int) or users <= 0:
                raise SystemExit(f"recorded {model_name}/{stage} user count is invalid")
            measured = {
                "quality_ndcg@10": float(quality),
                "quality_users": users,
            }
            scores[stage][model_name] = {
                **measured,
                "passes_provisional_gate": gates_pass(
                    measured, profile.contract.thresholds
                ),
                "runtime_seconds": None,
                "model": {
                    "source_sha256": source_hash,
                    "calibration_environment": environment,
                },
                "input_scope": ["environment/public/data"],
                "label_blind": True,
                "recorded": True,
            }
    return scores


def build_stage_reference_scores(
    thresholds: dict[str, float],
    stage: str,
    history_frames: dict[str, pd.DataFrame],
    history_end: int,
    catalog_df: pd.DataFrame,
    target_ids: set[int],
    seen: dict[int, set[int]],
    active_likes: dict[int, set[int]],
    graded: dict[int, dict[int, int]],
    replay_labels: dict[int, dict[int, int]] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Measure cutoff-correct shortcuts and the private reference model."""
    positive = history_frames["listens"][
        history_frames["listens"].played_ratio_pct >= LISTEN_POSITIVE
    ]
    catalog_items = catalog_df.sort_values("popularity_rank").item_id.to_numpy(
        dtype=np.int64
    )
    profile = build_profile(
        positive,
        target_ids,
        catalog_items,
        history_end=history_end,
    )

    popularity_ranker: Ranker = lambda uid: [
        int(item)
        for item in catalog_items
        if int(item) not in seen.get(int(uid), set())
    ][:TOP_N]

    expanded_similarity, expanded_ids = fit_similarity(
        positive,
        catalog_items,
        recent_days=REFERENCE_RECENT_DAYS,
        k=REFERENCE_K,
        history_end=history_end,
    )
    expanded_ranker = make_filtered_ranker(
        expanded_similarity, expanded_ids, profile, seen, catalog_items
    )

    active_like_rows = [
        (uid, history_end - 1, item)
        for uid, items in active_likes.items()
        for item in items
    ]
    active_like_events = pd.DataFrame(
        active_like_rows, columns=["uid", "timestamp", "item_id"]
    )
    oracle_similarity, oracle_ids = fit_similarity(
        positive,
        catalog_items,
        recent_days=REFERENCE_RECENT_DAYS,
        k=REFERENCE_K,
        history_end=history_end,
        organic_weight=2.0,
        like_events=active_like_events,
        like_weight=4.0,
    )
    oracle_ranker = make_filtered_ranker(
        oracle_similarity, oracle_ids, profile, seen, catalog_items
    )

    rankers: dict[str, Ranker] = {
        "popularity_on_eligible_candidates": popularity_ranker,
        "expanded_cf_without_signal_repair": expanded_ranker,
        "reference_organic_explicit_cf": oracle_ranker,
    }

    if replay_labels is not None:

        def validation_replay_filtered(uid: int) -> list[int]:
            known = seen.get(int(uid), set())
            gold = replay_labels.get(int(uid), {})
            ranked = [
                item
                for item, _grade in sorted(
                    gold.items(), key=lambda pair: (-pair[1], pair[0])
                )
                if item not in known
            ]
            selected = set(ranked)
            for raw_item in catalog_items:
                if len(ranked) >= TOP_N:
                    break
                item = int(raw_item)
                if item not in known and item not in selected:
                    ranked.append(item)
                    selected.add(item)
            return ranked[:TOP_N]

        rankers["validation_label_replay_after_submission_filter"] = (
            validation_replay_filtered
        )

    scores = {name: score_ranker(ranker, graded) for name, ranker in rankers.items()}
    for name, values in scores.items():
        values["passes_provisional_gate"] = gates_pass(values, thresholds)
        log(
            f"  {stage}/{name}: quality_ndcg@10={values['quality_ndcg@10']:.6f} "
            f"pass={values['passes_provisional_gate']}"
        )
    if any(
        scores[name]["passes_provisional_gate"]
        for name in rankers
        if name != "reference_organic_explicit_cf"
    ):
        raise SystemExit(f"a measured {stage} shortcut passes the review gate")
    if not scores["reference_organic_explicit_cf"]["passes_provisional_gate"]:
        raise SystemExit(f"the cutoff-correct {stage} reference does not pass")
    return scores


def build_stage_shortcut_scores(
    thresholds: dict[str, float],
    stage: str,
    catalog_df: pd.DataFrame,
    seen: dict[int, set[int]],
    graded: dict[int, dict[int, int]],
    replay_labels: dict[int, dict[int, int]] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Measure non-model shortcuts without treating them as the reference."""
    catalog_items = catalog_df.sort_values("popularity_rank").item_id.to_numpy(
        dtype=np.int64
    )

    popularity_ranker: Ranker = lambda uid: [
        int(item)
        for item in catalog_items
        if int(item) not in seen.get(int(uid), set())
    ][:TOP_N]
    rankers: dict[str, Ranker] = {
        "popularity_on_eligible_candidates": popularity_ranker,
    }

    if replay_labels is not None:

        def validation_replay_filtered(uid: int) -> list[int]:
            known = seen.get(int(uid), set())
            gold = replay_labels.get(int(uid), {})
            ranked = [
                item
                for item, _grade in sorted(
                    gold.items(), key=lambda pair: (-pair[1], pair[0])
                )
                if item not in known
            ]
            selected = set(ranked)
            for raw_item in catalog_items:
                if len(ranked) >= TOP_N:
                    break
                item = int(raw_item)
                if item not in known and item not in selected:
                    ranked.append(item)
                    selected.add(item)
            return ranked[:TOP_N]

        rankers["validation_label_replay_after_submission_filter"] = (
            validation_replay_filtered
        )

    scores = {name: score_ranker(ranker, graded) for name, ranker in rankers.items()}
    for name, values in scores.items():
        values["passes_provisional_gate"] = gates_pass(values, thresholds)
        log(
            f"  {stage}/{name}: quality_ndcg@10="
            f"{values['quality_ndcg@10']:.6f} "
            f"pass={values['passes_provisional_gate']}"
        )
    if any(scores[name]["passes_provisional_gate"] for name in rankers):
        raise SystemExit(f"a measured {stage} shortcut passes the review gate")
    return scores


def run_json_process(
    command: list[str], *, cwd: Path, description: str
) -> dict[str, object]:
    """Run one deterministic build-time check and parse its JSON stdout."""
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            f"{description} failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit(f"{description} returned invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{description} returned a non-object JSON payload")
    return payload


def run_private_two_tower_model(
    profile: TaskProfile,
    out: Path,
    *,
    solver_name: str,
    anchor_name: str,
) -> dict[str, dict[str, dict[str, object]]]:
    """Fit and record one public-data-only two-tower implementation."""
    scratch = out / f"tests/private/.{anchor_name}"
    validation_output = scratch / "validation.parquet"
    solution = scratch / "solution"
    submission_output = solution / "submission.parquet"
    scratch.mkdir(parents=True)
    solution.mkdir()
    (solution / "README.md").write_text(
        "Calibration fixture for the two-tower negative-sampling improvement. "
        "This fixture is generated from public history only and records "
        "platform-local production and reference scores. Authoritative launch "
        "gates are calibrated in the Harbor CPU containers.\n",
        encoding="utf-8",
    )
    solver = (out / "tests/private" / solver_name).resolve()
    public = (out / "environment/public").resolve()

    try:
        validation_run = run_json_process(
            [
                sys.executable,
                str(solver),
                "--stage",
                "validation",
                "--public-dir",
                str(public),
                "--output",
                str(validation_output.resolve()),
            ],
            cwd=solver.parent,
            description=f"private {anchor_name} validation build",
        )
        public_report = run_json_process(
            [
                sys.executable,
                str((out / "environment/public/evaluate_public.py").resolve()),
                str(validation_output.resolve()),
            ],
            cwd=(out / "environment/public").resolve(),
            description=f"private {anchor_name} public evaluation",
        )
        public_scores = dict(public_report.get("scores", {}))
        public_passes = gates_pass(public_scores, profile.contract.thresholds)

        submission_run = run_json_process(
            [
                sys.executable,
                str(solver),
                "--stage",
                "submission",
                "--public-dir",
                str(public),
                "--output",
                str(submission_output.resolve()),
            ],
            cwd=solver.parent,
            description=f"private {anchor_name} submission build",
        )
        hidden_command = [
            sys.executable,
            str((out / "tests/evaluate_submission.py").resolve()),
            str(solution.resolve()),
        ]
        hidden_report = run_json_process(
            hidden_command,
            cwd=(out / "tests").resolve(),
            description=f"private {anchor_name} hidden evaluation",
        )
        try:
            validate_verifier_result(hidden_report, expected_task_id=profile.task_id)
        except VerifierResultError as error:
            raise SystemExit(
                f"{anchor_name} returned an invalid verifier result: {error}"
            ) from error
        if hidden_report.get("status") != "scored":
            raise SystemExit(f"the private judge did not score {anchor_name}")
        hidden_scores = dict(hidden_report.get("metrics", {}))
        hidden_passes = gates_pass(hidden_scores, profile.contract.thresholds)

        validation_model = {
            "architecture": validation_run.get("architecture"),
            "config": validation_run.get("config"),
        }
        submission_model = {
            "architecture": submission_run.get("architecture"),
            "config": submission_run.get("config"),
        }
        if validation_model != submission_model:
            raise SystemExit(f"{anchor_name} configuration changed between stages")

        anchors: dict[str, dict[str, dict[str, object]]] = {
            "public_validation": {
                anchor_name: {
                    **public_scores,
                    "passes_provisional_gate": public_passes,
                    "runtime_seconds": validation_run.get("runtime_seconds"),
                    "model": validation_model,
                    "input_scope": ["environment/public/data"],
                    "label_blind": True,
                }
            },
            "hidden_test": {
                anchor_name: {
                    **hidden_scores,
                    "passes_provisional_gate": hidden_passes,
                    "runtime_seconds": submission_run.get("runtime_seconds"),
                    "model": submission_model,
                    "input_scope": ["environment/public/data"],
                    "label_blind": True,
                }
            },
        }
        log(
            f"  public_validation/{anchor_name}: "
            f"quality_ndcg@10={public_scores['quality_ndcg@10']:.6f} "
            f"pass={public_passes}"
        )
        log(
            f"  hidden_test/{anchor_name}: "
            f"quality_ndcg@10={hidden_scores['quality_ndcg@10']:.6f} "
            f"pass={hidden_passes}"
        )
        return anchors
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def publish_staged_package(staged_out: Path, final_out: Path) -> None:
    """Replace ``final_out`` only after its staged replacement is complete."""
    previous_out = final_out.parent / f".{final_out.name}.previous"
    if previous_out.exists() or previous_out.is_symlink():
        raise RuntimeError(f"refusing to overwrite recovery package at {previous_out}")
    had_previous = final_out.exists() or final_out.is_symlink()
    if had_previous:
        final_out.replace(previous_out)
    try:
        staged_out.replace(final_out)
    except BaseException:
        if had_previous:
            try:
                previous_out.replace(final_out)
            except OSError as rollback_error:
                raise RuntimeError(
                    f"failed to publish {staged_out} and restore {final_out}; "
                    f"the previous package remains at {previous_out}"
                ) from rollback_error
        raise
    if had_previous:
        if previous_out.is_dir() and not previous_out.is_symlink():
            shutil.rmtree(previous_out)
        else:
            previous_out.unlink()


def write_static_task_sources(profile: TaskProfile, out: Path) -> None:
    """Materialize all non-dataset files needed by a standalone package."""
    variant_dir = out / "environment/public/variant"
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / ".keep").write_text("", encoding="utf-8")
    write_task_brief(profile, out)
    write_task_runtime_contract(profile, out)
    write_public_evaluator(out)
    write_public_current_model(profile, out)
    write_private_reference_solver(profile, out)
    write_private_readme(profile, out)
    write_private_evaluator(profile, out)


def build_code_only_task(profile: TaskProfile, final_out: Path) -> None:
    """Publish a validated Harbor task skeleton without touching dataset state."""
    final_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{profile.task_id}.code-only-",
        dir=final_out.parent,
        ignore_cleanup_errors=True,
    ) as staging_directory:
        staged_out = Path(staging_directory) / profile.task_id
        for relative in (
            "environment/public/data",
            "environment/public/upstream",
            "environment/public/variant",
            "tests/private",
            "tests/public/data",
            "solution",
        ):
            (staged_out / relative).mkdir(parents=True)
        write_static_task_sources(profile, staged_out)
        run(
            [
                sys.executable,
                str(REPO_ROOT / "rl-environment/scripts/validate_task_package.py"),
                str(staged_out),
            ]
        )
        publish_staged_package(staged_out, final_out)


def main(profile: TaskProfile, argv: list[str] | None = None) -> int:
    rerun_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--out", type=Path, default=profile.default_out)
    parser.add_argument("--env-dir", type=Path, default=profile.default_env_dir)
    parser.add_argument(
        "--use-current-python",
        action="store_true",
        help="internal: build without creating or syncing an environment",
    )
    parser.add_argument(
        "--force-env",
        action="store_true",
        help="recreate and resync the task environment",
    )
    parser.add_argument(
        "--skip-env-sync",
        action="store_true",
        help="use the current Python without creating or syncing an environment",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="redownload source parquet files before building",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="use existing source parquet files",
    )
    parser.add_argument(
        "--skip-validate-source",
        action="store_true",
        help="do not inspect source parquet schemas before building",
    )
    parser.add_argument(
        "--skip-package-validation",
        action="store_true",
        help="build without running validate_task_package.py",
    )
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="build only the data-free Harbor task skeleton",
    )
    args = parser.parse_args(argv)
    args.source_dir = resolve_repo_path(args.source_dir)
    args.out = resolve_repo_path(args.out)
    args.env_dir = resolve_repo_path(args.env_dir)

    if args.force_download and args.skip_download:
        raise SystemExit("choose only one of --force-download and --skip-download")
    if args.force_env and args.skip_env_sync:
        raise SystemExit("choose only one of --force-env and --skip-env-sync")
    if args.out.name != profile.task_id:
        raise SystemExit(f"--out directory name must be {profile.task_id!r}")

    if args.code_only:
        incompatible = (
            args.use_current_python
            or args.force_env
            or args.skip_env_sync
            or args.force_download
            or args.skip_download
            or args.skip_validate_source
            or args.skip_package_validation
        )
        if incompatible:
            raise SystemExit("--code-only cannot be combined with full-build flags")
        build_code_only_task(profile, args.out)
        log(f"code-only task complete: {args.out}")
        return 0

    if not args.use_current_python and not args.skip_env_sync:
        python = sync_environment(profile, args.env_dir, force=args.force_env)
        rerun_in_environment(profile, python, args.env_dir, rerun_argv)

    require_python_version(profile)
    load_build_dependencies()
    if not args.skip_download:
        prepare_source_data(
            args.source_dir,
            force=args.force_download,
            validate=not args.skip_validate_source,
        )
    require_sources(args.source_dir)
    verify_production_model_delta(profile)

    final_out = args.out
    final_out.parent.mkdir(parents=True, exist_ok=True)
    staging_workspace = tempfile.TemporaryDirectory(
        prefix=f".{profile.task_id}.build-",
        dir=final_out.parent,
        ignore_cleanup_errors=True,
    )
    args.out = Path(staging_workspace.name) / profile.task_id
    log(f"staging package at {args.out}")
    for relative in (
        "environment/public/data",
        "environment/public/upstream",
        "environment/public/variant",
        "tests/private",
        "tests/public/data",
        "solution",
    ):
        (args.out / relative).mkdir(parents=True)

    write_static_task_sources(profile, args.out)
    if profile.is_case:
        run(
            [
                sys.executable,
                str(
                    (
                        args.out / "tests/private/test_shared_negative_pool.py"
                    ).resolve()
                ),
            ],
            cwd=(args.out / "tests/private").resolve(),
        )

    sources = load_sources(args.source_dir)
    splits = {name: split_frame(frame) for name, frame in sources.items()}
    train_frames = {name: values[0] for name, values in splits.items()}
    val_frames = {name: values[1] for name, values in splits.items()}
    test_frames = {name: values[2] for name, values in splits.items()}
    submission_history = {
        name: pd.concat([train_frames[name], val_frames[name]], ignore_index=True)
        for name in EVENT_NAMES
    }

    catalog_df, targets_df, catalog, target_ids = build_catalog_and_targets(
        train_frames["listens"]
    )
    write_public_data(args.out, splits, catalog_df, targets_df)
    write_verifier_public_inputs(args.out)
    write_upstream_sources(args.out)
    run_public_eligibility_build(args.out)

    log("4/11 materialising cutoff-specific state and encounter checks")
    validation_seen = build_seen(train_frames, target_ids, catalog)
    submission_seen = build_seen(submission_history, target_ids, catalog)
    verify_upstream_eligibility(
        args.out, "validation", validation_seen, catalog_df, targets_df
    )
    verify_upstream_eligibility(
        args.out, "submission", submission_seen, catalog_df, targets_df
    )
    finalise_public_candidate_inputs(args.out)
    encounter_codes = flatten_seen_codes(submission_seen)
    np.save(args.out / "tests/private/encounter_codes.npy", encounter_codes)
    validation_state, _, _ = build_expected_state(train_frames, target_ids, catalog)
    submission_state, _, _ = build_expected_state(
        submission_history, target_ids, catalog
    )
    validation_state.to_parquet(
        args.out / "environment/public/data/validation_preference_state.parquet",
        index=False,
        compression="zstd",
    )
    submission_state.to_parquet(
        args.out / "environment/public/data/test_preference_state.parquet",
        index=False,
        compression="zstd",
    )

    log("5/11 building hidden-test truth")
    validation_graded = build_truth(
        val_frames["listens"],
        val_frames["likes"],
        validation_seen,
        target_ids,
        catalog,
    )
    test_graded = build_truth(
        test_frames["listens"],
        test_frames["likes"],
        submission_seen,
        target_ids,
        catalog,
    )
    (args.out / "tests/private/truth.json").write_text(
        json.dumps(serialise_truth(test_graded), separators=(",", ":")),
        encoding="utf-8",
    )
    anti_cheat_report = verify_temporal_isolation(
        args.out,
        validation_seen,
        submission_seen,
        validation_graded,
        test_graded,
    )

    run_public_evaluator_smoke_check(args.out)

    log("7/11 checking task brief and judge")
    validation_shortcut_scores = build_stage_shortcut_scores(
        profile.contract.thresholds,
        "public_validation",
        catalog_df,
        validation_seen,
        validation_graded,
    )
    hidden_shortcut_scores = build_stage_shortcut_scores(
        profile.contract.thresholds,
        "hidden_test",
        catalog_df,
        submission_seen,
        test_graded,
        replay_labels=validation_graded,
    )
    if profile.is_case:
        two_tower_scores = load_recorded_case_model_scores(profile)
    else:
        two_tower_scores = run_private_two_tower_model(
            profile,
            args.out,
            solver_name="reference_solver.py",
            anchor_name="pytorch_windowed_two_tower",
        )
    reference_scores = {
        "public_validation": {
            **validation_shortcut_scores,
            **two_tower_scores["public_validation"],
        },
        "hidden_test": {
            **hidden_shortcut_scores,
            **two_tower_scores["hidden_test"],
        },
    }
    log("8/11 running adversarial judge checks")
    run_adversarial_judge_checks(
        profile,
        args.out,
        targets_df,
        catalog_df,
        submission_seen,
        validation_graded,
        anti_cheat_report,
    )

    log("9/11 writing evaluator-owned reports")
    (args.out / "tests/private/reference_scores.json").write_text(
        json.dumps(reference_scores, indent=2) + "\n", encoding="utf-8"
    )
    write_anti_cheat_report(args.out, anti_cheat_report)

    log("10/11 removing generated caches")
    for cache in args.out.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    if not args.skip_package_validation:
        log("11/11 validating task package")
        run(
            [
                sys.executable,
                str(REPO_ROOT / "rl-environment/scripts/validate_task_package.py"),
                str(args.out),
            ]
        )

    publish_staged_package(args.out, final_out)
    staging_workspace.cleanup()
    args.out = final_out
    log("build complete")
    log(f"  task={args.out}")
    log(f"  targets={len(targets_df):,} catalog={len(catalog_df):,}")
    log(
        f"  validation_state_rows={len(validation_state):,} "
        f"test_state_rows={len(submission_state):,} "
        f"hidden_quality_users={len(test_graded):,} "
        f"encounter_codes={len(encounter_codes):,}"
    )
    return 0
