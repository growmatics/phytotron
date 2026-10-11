"""Regenerate the small golden run: seven records (five seasons, two failures) and their summary.

The records are synthetic. Three days of 15-minute steps, two arms, and traces chosen so
that every crop-rule kind in ``configs/benchmark.yaml`` is broken by at least one season,
one season carries no climate block (a record from before the traces existed), one record
is a controller failure with an attempt history, and one is a season the harness could
not start.

``tests/test_golden.py`` summarises the stored records and compares the result with the
stored ``summary.json``. Run this script only in a commit that intends to change what the
summary looks like, and say so in the commit message:

    .venv/bin/python tests/fixtures/run-small/regenerate.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from phytotron.config import load_config
from phytotron.envs.glg2 import SeasonResult
from phytotron.grid import Task
from phytotron.records import SEASON_KEYS, Attempt, FailureRecord, Record, season_record
from phytotron.summary import summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = ROOT / "configs" / "benchmark.yaml"

DAYS = 3
STEP_SECONDS = 900
STEPS = DAYS * 86400 // STEP_SECONDS + 1  # the runner records one state more than steps


def _climate(temperature: float, humidity: float) -> dict[str, np.ndarray]:
    return {
        "temperature": np.full(STEPS, temperature),
        "humidity": np.full(STEPS, humidity),
        "radiation": np.zeros(STEPS),
    }


def _season(
    arm: str,
    year: int,
    mismatch: float,
    seed: int,
    *,
    net: float,
    climate: dict[str, np.ndarray] | None,
    humidity_exceedance: float = 0.0,
) -> Record:
    task = Task(arm, year, mismatch, seed, DAYS)
    mpc = arm == "physics_mpc"
    result = SeasonResult(
        net_economic_value=net,
        exceedances={
            "temperature": np.zeros(STEPS),
            "co2": np.zeros(STEPS),
            "humidity": np.full(STEPS, humidity_exceedance),
            "lamp": np.zeros(STEPS),
        },
        online_cost_seconds=0.19 if mpc else 0.0002,
        solve_success=0.97 if mpc else None,
        revenue=net + 4.0,
        heating_cost=2.5,
        co2_cost=0.5,
        electricity_cost=1.0,
        steps=STEPS,
        solve_counts={"ok": 280, "max_iter": 8} if mpc else None,
        max_action_excess=0.0,
        climate=climate,
    )
    return replace(
        season_record(task, result),
        started="2026-10-11T08:00:00+00:00",
        finished="2026-10-11T08:00:19+00:00",
        wall_seconds=19.0,
        attempts=(),
        keys=(*SEASON_KEYS, "started", "finished", "wall_seconds", "attempts"),
    )


def _failure(
    arm: str, year: int, mismatch: float, seed: int, *, stage: str, error: str, retried: bool
) -> Record:
    task = Task(arm, year, mismatch, seed, DAYS)
    attempt = Attempt(error, stage, "2026-10-11T07:30:40+00:00")
    return FailureRecord(
        task=task,
        label=task.label,
        error=error,
        stage=stage,
        failed_at="2026-10-11T08:00:40+00:00",
        traceback="Traceback (most recent call last):\n  ...\n" + error + "\n",
        attempts=(attempt,) if retried else (),
        keys=("task", "label", "error", "stage", "failed_at", "traceback", "attempts"),
    )


def build_records() -> dict[str, Record]:
    """The seven records, keyed by task key, in a fixed order."""
    calm = _climate(18.0, 80.0)
    hot_and_wet = _climate(35.0, 95.0)  # breaks every temperature and humidity rule
    cold = _climate(8.0, 80.0)  # breaks the cold side of peak, dose and the 6-day mean
    records = [
        _season("incumbent", 2024, 0.0, 0, net=2.0, climate=calm),
        _season("incumbent", 2025, 0.0, 0, net=1.5, climate=None),  # older record
        _season("physics_mpc", 2024, 0.0, 0, net=-1.0, climate=calm),
        _season(
            "physics_mpc", 2024, 0.1, 0, net=-3.0, climate=hot_and_wet, humidity_exceedance=10.0
        ),
        # the harness could not start this one: listed, not counted
        _failure(
            "physics_mpc",
            2024,
            0.1,
            1,
            stage="setup",
            error="FileNotFoundError: weather file for 2024",
            retried=False,
        ),
        _season("physics_mpc", 2025, 0.1, 0, net=-2.0, climate=cold),
        # the controller failed twice during the season: counted against its gate
        _failure(
            "physics_mpc",
            2025,
            0.1,
            1,
            stage="season",
            error="RuntimeError: solver returned no solution",
            retried=True,
        ),
    ]
    return {r.task.key: r for r in records}


def run_record(records: dict[str, Record]) -> dict[str, Any]:
    return {
        "timestamp": "2026-10-11T08:00:00+00:00",
        "git_commit": "fixture",
        "git_dirty": False,
        "config_sha256": "fixture",
        "season_days": DAYS,
        "workers": 1,
        "grid": list(records),
    }


def write() -> None:
    cfg = load_config(CONFIG)
    records = build_records()
    seasons = HERE / "seasons"
    seasons.mkdir(exist_ok=True)
    for old in seasons.glob("*.json"):
        old.unlink()
    for key, record in records.items():
        (seasons / f"{key}.json").write_text(json.dumps(record.to_json(), indent=1) + "\n")
    launch = run_record(records)
    (HERE / "run_record.json").write_text(json.dumps(launch, indent=1) + "\n")
    summary = summarize(records, cfg, launch)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {len(records)} records and summary.json to {HERE}")


if __name__ == "__main__":
    write()
