"""Tests for the season records on disk: types, round trips and the attempt history."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from synthetic import _calm, _record

from phytotron.config import load_config
from phytotron.grid import Task
from phytotron.records import (
    Attempt,
    FailureRecord,
    SeasonRecord,
    load_records,
    record_from_json,
    save_record,
    with_attempts,
)
from phytotron.summary import season_row

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "benchmark.yaml"
CFG = load_config(CONFIG)
GOLDEN_SEASONS = ROOT / "tests" / "fixtures" / "run-small" / "seasons"
RUN_2_SEASONS = ROOT / "runs" / "full-2" / "seasons"  # not in the repository


def test_record_round_trips_through_disk(tmp_path: Path) -> None:
    task = Task("physics_mpc", 2025, 0.1, 0, 60)
    record = _record("physics_mpc", 0.1, 0, -1.5, [0.0, 3.0])
    save_record(tmp_path, task, record)
    loaded = load_records(tmp_path)
    assert list(loaded) == [task.key]
    assert loaded[task.key] == record
    season = loaded[task.key].result()
    assert season.net_economic_value == -1.5
    assert season.heating_cost == 1.0
    assert season.steps == 2
    np.testing.assert_array_equal(season.exceedances["humidity"], [0.0, 3.0])


def _without(record: SeasonRecord, *names: str) -> SeasonRecord:
    """The record as an older run wrote it, before ``names`` existed."""
    gone = dict.fromkeys(names)
    return replace(record, keys=tuple(k for k in record.keys if k not in names), **gone)


def test_records_without_breakdown_still_load() -> None:
    record = _without(
        _record("physics_mpc", 0.1, 0, -1.5, [0.0]),
        "revenue",
        "heating_cost",
        "co2_cost",
        "electricity_cost",
        "steps",
    )
    assert "revenue" not in record.to_json()
    assert record.result().revenue is None
    assert season_row(record, CFG)["fruit_growth_kg_fw"] is None


def test_attempt_history_keeps_each_failure() -> None:
    task = Task("physics_mpc", 2025, 0.1, 0, 60)
    first = FailureRecord(task, task.label, "RuntimeError: x", stage="season", failed_at="t1")
    second = with_attempts(replace(first, failed_at="t2"), first)
    assert second.attempts == (Attempt("RuntimeError: x", "season", "t1"),)
    assert second.to_json()["attempts"] == [
        {"error": "RuntimeError: x", "stage": "season", "failed_at": "t1"}
    ]
    final = with_attempts(_record("physics_mpc", 0.1, 0, 1.0, [0.0]), second)
    assert [a.failed_at for a in final.attempts] == ["t1", "t2"]
    assert final.keys[-1] == "attempts"
    assert with_attempts(_record("physics_mpc", 0.1, 0, 1.0, [0.0]), None).attempts == ()


def test_climate_traces_round_trip_and_old_records_load() -> None:
    record = _record("physics_mpc", 0.1, 0, -1.0, [0.0], climate=_calm(2))
    season = record.result()
    assert season.climate is not None
    np.testing.assert_array_equal(season.climate["temperature"], [18.0, 18.0])
    old = _without(_record("physics_mpc", 0.1, 0, -1.0, [0.0]), "climate")
    assert old.result().climate is None
    assert season_row(old, CFG)["broken_rules"] is None
    assert season_row(old, CFG)["night_humidity_deficit_g_m3"] is None


def test_failure_keeps_a_checked_season_and_counts_by_stage() -> None:
    kept = _record("physics_mpc", 0.1, 0, -1.0, [0.0])
    failure = FailureRecord.from_json(
        {**kept.to_json(), "error": "SeasonCheckError: ran 1 of 2 steps", "stage": "check"}
    )
    assert failure.season == kept and failure.counted
    assert not FailureRecord.from_json({**kept.to_json(), "error": "x", "stage": "setup"}).counted
    assert FailureRecord.from_json(
        {"task": kept.to_json()["task"], "label": "l", "error": "x"}
    ).counted
    assert failure.to_json() == {
        **kept.to_json(),
        "error": "SeasonCheckError: ran 1 of 2 steps",
        "stage": "check",
    }


@pytest.mark.parametrize("path", sorted(GOLDEN_SEASONS.glob("*.json")), ids=lambda p: p.stem)
def test_every_golden_record_writes_back_byte_for_byte(path: Path) -> None:
    text = path.read_text()
    assert json.dumps(record_from_json(json.loads(text)).to_json(), indent=1) + "\n" == text


@pytest.mark.skipif(
    not RUN_2_SEASONS.is_dir(), reason="run 2 season records are kept outside the repository"
)
def test_every_run_2_record_writes_back_byte_for_byte() -> None:
    # Records from before the attempts, stages and user-controller fields existed must
    # keep their own key set: to_json omits what the file never had.
    for path in sorted(RUN_2_SEASONS.glob("*.json")):
        text = path.read_text()
        assert json.dumps(record_from_json(json.loads(text)).to_json()) == text, path.name
