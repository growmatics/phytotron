"""Season records: one JSON file per season in a run directory, typed.

A completed season is a :class:`SeasonRecord`, written with the full per-step traces; a
failed season is a :class:`FailureRecord` with its error, the stage it stopped at (see
:attr:`FailureRecord.counted` for which stages count against the arm's gate), the attempt
history of the record it replaces, and, for a season that failed its checks, the season
data it kept. ``to_json`` reproduces a record's file exactly: today's keys in today's
order for a new record, and only the keys an older record had, in its order, for one
read from disk, so every stored file round-trips byte for byte.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from phytotron.envs.glg2 import SeasonResult
from phytotron.grid import Task

BREAKDOWN_FIELDS = ("revenue", "heating_cost", "co2_cost", "electricity_cost")
EXTRA_FIELDS = ("steps", "solve_counts", "max_action_excess")  # optional in older records
STAGES = ("setup", "build", "season", "check")  # where a failed season stopped
TASK_KEYS = tuple(f.name for f in fields(Task))
# A new season record's keys, in file order.
SEASON_KEYS = (
    "task",
    "label",
    "net_economic_value",
    "online_cost_seconds",
    "solve_success",
    *BREAKDOWN_FIELDS,
    *EXTRA_FIELDS,
    "exceedances",
    "climate",
)
FAILURE_KEYS = ("error", "stage", "failed_at", "traceback")


@dataclass(frozen=True)
class Attempt:
    """One earlier, failed attempt at a season: its error, stage and time."""

    error: str | None
    stage: str | None
    failed_at: str | None

    def to_json(self) -> dict[str, Any]:
        return {"error": self.error, "stage": self.stage, "failed_at": self.failed_at}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Attempt:
        return cls(
            error=data.get("error"), stage=data.get("stage"), failed_at=data.get("failed_at")
        )


def task_json(task: Task, keys: tuple[str, ...]) -> dict[str, Any]:
    """A task as plain data, the given fields in the given order."""
    return {k: getattr(task, k) for k in keys}


@dataclass(frozen=True)
class SeasonRecord:
    """A completed season as stored: its task, scalars, traces, timing and attempts."""

    task: Task
    label: str
    net_economic_value: float
    online_cost_seconds: float
    solve_success: float | None
    revenue: float | None = None
    heating_cost: float | None = None
    co2_cost: float | None = None
    electricity_cost: float | None = None
    steps: int | None = None
    solve_counts: dict[str, int] | None = None
    max_action_excess: float | None = None
    exceedances: dict[str, list[float]] = field(default_factory=dict)
    climate: dict[str, list[float]] | None = None
    started: str | None = None
    finished: str | None = None
    wall_seconds: float | None = None
    attempts: tuple[Attempt, ...] = ()
    # The keys this record carries, in file order (a record from before a field existed
    # keeps its own key set, so it writes back byte for byte).
    keys: tuple[str, ...] = SEASON_KEYS
    task_keys: tuple[str, ...] = TASK_KEYS

    def result(self) -> SeasonResult:
        """The season as the aggregation uses it, traces as arrays."""
        return SeasonResult(
            net_economic_value=float(self.net_economic_value),
            exceedances={k: np.asarray(v, dtype=float) for k, v in self.exceedances.items()},
            online_cost_seconds=float(self.online_cost_seconds),
            solve_success=self.solve_success,
            **{name: getattr(self, name) for name in (*BREAKDOWN_FIELDS, *EXTRA_FIELDS)},
            climate=None
            if self.climate is None
            else {k: np.asarray(v, dtype=float) for k, v in self.climate.items()},
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in self.keys:
            if key == "task":
                out[key] = task_json(self.task, self.task_keys)
            elif key == "attempts":
                out[key] = [a.to_json() for a in self.attempts]
            else:
                out[key] = getattr(self, key)
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SeasonRecord:
        scalars = {
            name: data.get(name)
            for name in (
                *BREAKDOWN_FIELDS,
                *EXTRA_FIELDS,
                "climate",
                "started",
                "finished",
                "wall_seconds",
            )
        }
        return cls(
            task=Task(**data["task"]),
            label=data["label"],
            net_economic_value=data["net_economic_value"],
            online_cost_seconds=data["online_cost_seconds"],
            solve_success=data["solve_success"],
            exceedances=dict(data["exceedances"]),
            attempts=tuple(Attempt.from_json(a) for a in data.get("attempts", [])),
            keys=tuple(data),
            task_keys=tuple(data["task"]),
            **scalars,
        )


@dataclass(frozen=True)
class FailureRecord:
    """A season that failed: where and why, what it kept, and its earlier attempts."""

    task: Task
    label: str
    error: str
    stage: str | None = None
    failed_at: str | None = None
    traceback: str | None = None
    attempts: tuple[Attempt, ...] = ()
    season: SeasonRecord | None = None  # a season that failed its checks keeps its data
    keys: tuple[str, ...] = ("task", "label", *FAILURE_KEYS)
    task_keys: tuple[str, ...] = TASK_KEYS

    @property
    def counted(self) -> bool:
        """Whether this failure counts against the controller's gate.

        A season that failed at stage ``setup`` (config, environment, reset) never reached
        the controller: it is listed as not started and not counted. A failure while the
        controller was built, during the season or in its checks is the controller's. A
        failure record without a stage (written before stages existed) counts.
        """
        return self.stage != "setup"

    def to_json(self) -> dict[str, Any]:
        kept = self.season.to_json() if self.season is not None else {}
        out: dict[str, Any] = {}
        for key in self.keys:
            if key == "task":
                out[key] = task_json(self.task, self.task_keys)
            elif key == "attempts":
                out[key] = [a.to_json() for a in self.attempts]
            elif key in ("label", *FAILURE_KEYS):
                out[key] = getattr(self, key)
            else:
                out[key] = kept[key]
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FailureRecord:
        own = {"attempts", *FAILURE_KEYS}
        kept = {k: v for k, v in data.items() if k not in own}
        season = SeasonRecord.from_json(kept) if "net_economic_value" in data else None
        return cls(
            task=Task(**data["task"]),
            label=data["label"],
            error=data["error"],
            stage=data.get("stage"),
            failed_at=data.get("failed_at"),
            traceback=data.get("traceback"),
            attempts=tuple(Attempt.from_json(a) for a in data.get("attempts", [])),
            season=season,
            keys=tuple(data),
            task_keys=tuple(data["task"]),
        )


Record = SeasonRecord | FailureRecord


def record_from_json(data: dict[str, Any]) -> Record:
    """A stored record as its type: a failure if it carries an error, else a season."""
    return FailureRecord.from_json(data) if "error" in data else SeasonRecord.from_json(data)


def season_record(task: Task, result: SeasonResult) -> SeasonRecord:
    """A season result as a record, with the full per-step traces as lists."""
    return SeasonRecord(
        task=task,
        label=task.label,
        net_economic_value=float(result.net_economic_value),
        online_cost_seconds=float(result.online_cost_seconds),
        solve_success=result.solve_success,
        **{name: getattr(result, name) for name in (*BREAKDOWN_FIELDS, *EXTRA_FIELDS)},
        exceedances={k: np.asarray(v, dtype=float).tolist() for k, v in result.exceedances.items()},
        climate=None
        if result.climate is None
        else {k: np.asarray(v, dtype=float).round(4).tolist() for k, v in result.climate.items()},
    )


def with_attempts(record: Record, previous: Record | None) -> Record:
    """``record`` with the attempt history of the record it replaces (a failed earlier run).

    Each failed attempt is kept as its error, stage and time, so a retried season shows
    that it failed before.
    """
    history = list(previous.attempts) if previous is not None else []
    if isinstance(previous, FailureRecord):
        history.append(Attempt(previous.error, previous.stage, previous.failed_at))
    keys = record.keys if "attempts" in record.keys else (*record.keys, "attempts")
    return replace(record, attempts=tuple(history), keys=keys)


def save_record(run_dir: Path, task: Task, record: Record) -> Path:
    """Write one season record to ``run_dir/seasons/<key>.json`` (atomic rename)."""
    path = run_dir / "seasons" / f"{task.key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record.to_json()))
    tmp.replace(path)
    return path


def load_records(run_dir: Path) -> dict[str, Record]:
    """All season records in a run directory, keyed by task key."""
    folder = run_dir / "seasons"
    if not folder.is_dir():
        return {}
    return {
        p.stem: record_from_json(json.loads(p.read_text())) for p in sorted(folder.glob("*.json"))
    }


def pending(
    tasks: Sequence[Task], done: Mapping[str, Record], *, retry_failed: bool = False
) -> list[Task]:
    """Tasks still to run: those without a record, plus failed ones if ``retry_failed``.

    A failed season stays failed on resume unless a retry is asked for, so a resumed
    run cannot quietly replace a crash with a lucky second attempt.
    """
    return [
        t
        for t in tasks
        if t.key not in done or (retry_failed and isinstance(done[t.key], FailureRecord))
    ]
