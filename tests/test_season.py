"""Tests for running one season: the stage of each failure and the checks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from synthetic import _Env, _fake_season, _record

from phytotron import season
from phytotron.config import load_config
from phytotron.envs.glg2 import action_excess, expected_steps
from phytotron.grid import Task
from phytotron.records import FailureRecord
from phytotron.season import Paths, run_task, run_task_safe, season_problems

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "benchmark.yaml"
CFG = load_config(CONFIG)


@pytest.mark.parametrize(
    ("broken", "stage"),
    [
        ("make_benchmark_env", "setup"),
        ("build_controller", "build"),
        ("run_season_controller", "season"),
    ],
)
def test_failed_season_records_the_stage_it_stopped_at(
    monkeypatch: pytest.MonkeyPatch, broken: str, stage: str
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(season, "make_benchmark_env", lambda *a, **k: _Env())
    monkeypatch.setattr(season, "build_controller", lambda *a, **k: object())
    monkeypatch.setattr(season, "run_season_controller", lambda *a, **k: _fake_season(97))
    monkeypatch.setattr(season, broken, boom)
    task = Task("incumbent", 2025, 0.0, 0, 1)
    record = run_task_safe(task, Paths(str(CONFIG), "", ""))
    assert isinstance(record, FailureRecord)
    assert record.error == "RuntimeError: boom"
    assert record.stage == stage
    assert record.task == task and record.season is None


def test_season_that_fails_its_checks_keeps_its_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(season, "make_benchmark_env", lambda *a, **k: _Env())
    monkeypatch.setattr(season, "build_controller", lambda *a, **k: object())
    monkeypatch.setattr(season, "run_season_controller", lambda *a, **k: _fake_season(50))
    task = Task("incumbent", 2025, 0.0, 0, 1)
    record = run_task_safe(task, Paths(str(CONFIG), "", ""))
    assert isinstance(record, FailureRecord)
    assert record.stage == "check" and "ran 50 of 97 steps" in record.error
    assert record.season is not None
    assert record.season.steps == 50 and record.season.net_economic_value == 1.0
    assert list(record.to_json())[-4:] == ["error", "stage", "failed_at", "traceback"]
    monkeypatch.setattr(season, "run_season_controller", lambda *a, **k: _fake_season(97))
    progress: dict = {}
    good = run_task(task, Paths(str(CONFIG), "", ""), progress)
    assert progress["stage"] == "check"
    assert good.wall_seconds is not None and good.wall_seconds >= 0.0
    assert good.started is not None and good.finished is not None and good.started <= good.finished
    assert list(good.to_json())[-3:] == ["started", "finished", "wall_seconds"]


def test_season_checks_catch_short_seasons_and_bad_actions() -> None:
    record = _record("physics_mpc", 0.1, 0, -1.0, [0.0])
    full = expected_steps(60, 900)
    assert full == 5761
    assert season_problems(replace(record, steps=full, max_action_excess=0.0), CFG) == []
    problems = season_problems(replace(record, steps=full - 10, max_action_excess=0.2), CFG)
    assert problems == [f"ran {full - 10} of {full} steps", "an action left its range by 0.2"]
    older = replace(record, steps=None, max_action_excess=None)  # nothing to check
    assert season_problems(older, CFG) == []


def test_action_excess_per_control_kind() -> None:
    assert action_excess(np.array([0.0, 1.0]), "absolute") == 0.0
    assert action_excess(np.array([-0.5, 1.2]), "absolute") == pytest.approx(0.5)
    assert action_excess(np.array([-1.0, 0.5]), "incremental") == 0.0
    assert action_excess(np.array([1.25]), "incremental") == pytest.approx(0.25)
