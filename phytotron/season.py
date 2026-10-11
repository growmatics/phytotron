"""Run one season of the grid: build the environment and the controller, run, check.

``run_task`` raises on any failure; ``run_task_safe`` turns a failure into an error record
with the stage it stopped at, so a crashed season is recorded, never dropped, and the grid
goes on. A finished season must have run all its steps with every action in range.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from phytotron.arms import (
    build_controller,
)
from phytotron.config import BenchmarkConfig, load_config
from phytotron.controllers.registry import build_registered
from phytotron.envs.glg2 import (
    expected_steps,
    make_benchmark_env,
    run_season_controller,
)
from phytotron.grid import Task
from phytotron.records import (
    FAILURE_KEYS,
    SEASON_KEYS,
    FailureRecord,
    Record,
    SeasonRecord,
    season_record,
)


class SeasonCheckError(RuntimeError):
    """A season ran but failed a runner check (too few steps, an action out of range)."""


@dataclass(frozen=True)
class Paths:
    """Where a worker finds the benchmark config, GreenLight-Gym2's configs, and weather."""

    benchmark_config: str
    glg2_config_dir: str
    weather_dir: str


def run_task(task: Task, paths: Paths, progress: dict[str, Any] | None = None) -> SeasonRecord:
    """Run one season and return it as a JSON-ready record, checked.

    ``progress["stage"]`` follows the run, so a caller can tell where a failure happened:
    ``setup`` (config, environment, reset: the harness's), ``build`` (the controller's
    constructor), ``season`` and ``check``. A season that fails a check raises
    :class:`SeasonCheckError` with the record attached.
    """
    progress = {} if progress is None else progress
    progress["stage"] = "setup"
    started, t0 = _utc_now(), time.perf_counter()
    cfg = load_config(paths.benchmark_config)
    env = make_benchmark_env(
        cfg,
        task.year,
        config_dir=paths.glg2_config_dir,
        weather_dir=paths.weather_dir,
        season_days=task.season_days,
    )
    env.reset(seed=task.seed)
    progress["stage"] = "build"
    controller = (
        build_registered(task.controller, env, cfg, mismatch=task.mismatch, seed=task.seed)
        if task.controller
        else build_controller(
            task.arm,
            env,
            cfg,
            glg2_config_dir=paths.glg2_config_dir,
            mismatch=task.mismatch,
            seed=task.seed,
        )
    )
    progress["stage"] = "season"
    record = replace(
        season_record(task, run_season_controller(env, controller, task.seed)),
        started=started,
        finished=_utc_now(),
        wall_seconds=time.perf_counter() - t0,
        keys=(*SEASON_KEYS, "started", "finished", "wall_seconds"),
    )
    progress["stage"] = "check"
    problems = season_problems(record, cfg)
    if problems:
        err = SeasonCheckError("; ".join(problems))
        err.record = record  # type: ignore[attr-defined]
        raise err
    return record


def season_problems(record: SeasonRecord, cfg: BenchmarkConfig) -> list[str]:
    """Why a finished season cannot be scored as a normal season (empty if it can).

    Too few steps means the simulator ended the season early (GreenLight-Gym2's gym step
    ends a season on an ODE error without raising). An action outside its range means the
    plant received controls it should not have (``step_raw_control`` does not clip).
    """
    problems = []
    want = expected_steps(record.task.season_days, cfg.episode.timestep_seconds)
    if record.steps is not None and record.steps != want:
        problems.append(f"ran {record.steps} of {want} steps")
    excess = record.max_action_excess
    if excess is not None and excess > cfg.checks.action_tolerance:
        problems.append(f"an action left its range by {excess:.3g}")
    return problems


def run_task_safe(task: Task, paths: Paths) -> Record:
    """Like :func:`run_task`, but a season that raises becomes an error record.

    A crashed season must be recorded, not dropped: the record says at which stage it
    failed (see :func:`counted_failure` for which stages count against the arm's gate),
    and a season that failed a check keeps its data. (A worker process that dies outright still
    stops the pool; rerunning the same command resumes the grid.)
    """
    progress: dict[str, Any] = {}
    try:
        return run_task(task, paths, progress)
    except Exception as exc:  # deliberately broad: isolate one season's failure
        kept: SeasonRecord | None = getattr(exc, "record", None)
        return FailureRecord(
            task=task,
            label=task.label,
            error=f"{type(exc).__name__}: {exc}".splitlines()[0][:200],
            stage=progress.get("stage", "setup"),
            failed_at=_utc_now(),
            traceback=traceback.format_exc(),
            season=kept,
            keys=(*(kept.keys if kept is not None else ("task", "label")), *FAILURE_KEYS),
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
