"""Tests for the benchmark grid, season records and the summary (no simulator needed)."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from phytotron import runner
from phytotron.config import load_config
from phytotron.controllers.registry import ArmSpec
from phytotron.envs.glg2 import SeasonResult, action_excess, expected_steps
from phytotron.metrics.safety import banded_violation
from phytotron.runner import (
    Paths,
    Task,
    _mean,
    arm_settings,
    build_tasks,
    file_sha256,
    input_hashes,
    load_records,
    package_versions,
    pending,
    rescore_records,
    run_task,
    run_task_safe,
    save_record,
    season_from_record,
    season_problems,
    season_record,
    season_row,
    seed_invariant,
    summarize,
    with_attempts,
)
from phytotron.scenarios import glg2_config_dir

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "benchmark.yaml"
CFG = load_config(CONFIG)


def test_grid_has_arm_aware_axes() -> None:
    tasks = build_tasks(CFG)
    incumbent = [t for t in tasks if t.arm == "incumbent"]
    mpc = [t for t in tasks if t.arm == "physics_mpc"]
    years = len(CFG.generalisation.heldout_years)
    levels = len(CFG.raw["mismatch"]["sweep_magnitudes"])
    seeds = int(CFG.raw["reporting"]["seeds_per_condition"])
    # The incumbent has no model to perturb and is deterministic: one season per year.
    assert len(incumbent) == years
    assert {t.mismatch for t in incumbent} == {0.0}
    # At mismatch 0 the MPC's model is the plant's for every seed: one seed per year.
    assert len(mpc) == years * (1 + (levels - 1) * seeds)
    assert {t.seed for t in mpc if t.mismatch == 0.0} == {0}
    assert all(t.season_days == CFG.generalisation.season_length_days for t in tasks)
    assert len({t.key for t in tasks}) == len(tasks)


def test_seed_invariance_rule() -> None:
    assert seed_invariant("incumbent", 0.0)
    assert seed_invariant("physics_mpc", 0.0)
    assert not seed_invariant("physics_mpc", 0.05)


def test_grid_overrides_and_unknown_arm() -> None:
    tasks = build_tasks(
        CFG, ["physics_mpc"], years=[2025], mismatch_levels=[0.0, 0.1], seeds=[0, 1], season_days=1
    )
    assert [(t.label, t.seed) for t in tasks] == [
        ("2025/m0.00", 0),
        ("2025/m0.10", 0),
        ("2025/m0.10", 1),
    ]
    assert {t.season_days for t in tasks} == {1}
    with pytest.raises(ValueError, match="unknown arms"):
        build_tasks(CFG, ["ppo"])


def test_label_and_key_formats_are_stable() -> None:
    # Run directories written by earlier versions are read back by these formats.
    task = Task("physics_mpc", 2022, 0.025, 3, 60)
    assert task.label == "2022/m0.03"
    assert task.key == "physics_mpc__2022__m0.025__s3"


def _record(  # noqa: PLR0917 - a compact test helper
    arm: str,
    level: float,
    seed: int,
    net: float,
    rh: list[float],
    year: int = 2025,
    climate: dict[str, list[float]] | None = None,
) -> dict:
    task = Task(arm, year, level, seed, 60)
    result = SeasonResult(
        net_economic_value=net,
        exceedances={
            "humidity": np.array(rh),
            "temperature": np.zeros(len(rh)),
            "co2": np.zeros(len(rh)),
        },
        online_cost_seconds=0.1 if arm == "physics_mpc" else 0.0,
        solve_success=1.0 if arm == "physics_mpc" else None,
        revenue=net + 1.6,
        heating_cost=1.0,
        co2_cost=0.4,
        electricity_cost=0.2,
        steps=len(rh),
        climate=None if climate is None else {k: np.array(v) for k, v in climate.items()},
    )
    return season_record(task, result)


def _calm(steps: int = 96) -> dict[str, list[float]]:
    return {"temperature": [18.0] * steps, "humidity": [80.0] * steps, "radiation": [0.0] * steps}


def test_record_round_trips_through_disk(tmp_path: Path) -> None:
    task = Task("physics_mpc", 2025, 0.1, 0, 60)
    record = _record("physics_mpc", 0.1, 0, -1.5, [0.0, 3.0])
    save_record(tmp_path, task, record)
    loaded = load_records(tmp_path)
    assert list(loaded) == [task.key]
    season = season_from_record(loaded[task.key])
    assert season.net_economic_value == -1.5
    assert season.heating_cost == 1.0
    assert season.steps == 2
    np.testing.assert_array_equal(season.exceedances["humidity"], [0.0, 3.0])


def test_records_without_breakdown_still_load() -> None:
    record = _record("physics_mpc", 0.1, 0, -1.5, [0.0])
    for name in ("revenue", "heating_cost", "co2_cost", "electricity_cost", "steps"):
        del record[name]
    season = season_from_record(record)
    assert season.revenue is None
    assert season_row(record, CFG)["fruit_growth_kg_fw"] is None


def test_pending_keeps_failures_unless_retry_is_asked() -> None:
    a, b, c = (Task("physics_mpc", 2025, 0.0, s, 60) for s in range(3))
    done = {a.key: {"net_economic_value": 1.0}, b.key: {"error": "boom"}}
    assert pending([a, b, c], done) == [c]
    assert pending([a, b, c], done, retry_failed=True) == [b, c]
    assert pending([a], {}) == [a]


def test_attempt_history_keeps_each_failure() -> None:
    first = {"error": "RuntimeError: x", "stage": "season", "failed_at": "t1", "task": {}}
    second = with_attempts({**first, "failed_at": "t2"}, first)
    expected = {"error": "RuntimeError: x", "stage": "season", "failed_at": "t1"}
    assert second["attempts"] == [expected]
    final = with_attempts({"net_economic_value": 1.0}, second)
    assert [a["failed_at"] for a in final["attempts"]] == ["t1", "t2"]
    assert with_attempts({"net_economic_value": 1.0}, None)["attempts"] == []


class _Env:
    def reset(self, seed: int = 0) -> None:
        self.seed = seed


def _fake_season(steps: int) -> SeasonResult:
    return SeasonResult(
        net_economic_value=1.0,
        exceedances={"humidity": np.zeros(steps), "temperature": np.zeros(steps)},
        online_cost_seconds=0.0,
        solve_success=None,
        steps=steps,
        max_action_excess=0.0,
    )


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

    monkeypatch.setattr(runner, "make_benchmark_env", lambda *a, **k: _Env())
    monkeypatch.setattr(runner, "build_controller", lambda *a, **k: object())
    monkeypatch.setattr(runner, "run_season_controller", lambda *a, **k: _fake_season(97))
    monkeypatch.setattr(runner, broken, boom)
    task = Task("incumbent", 2025, 0.0, 0, 1)
    record = run_task_safe(task, Paths(str(CONFIG), "", ""))
    assert record["error"] == "RuntimeError: boom"
    assert record["stage"] == stage
    assert record["task"] == asdict(task) and "net_economic_value" not in record


def test_season_that_fails_its_checks_keeps_its_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "make_benchmark_env", lambda *a, **k: _Env())
    monkeypatch.setattr(runner, "build_controller", lambda *a, **k: object())
    monkeypatch.setattr(runner, "run_season_controller", lambda *a, **k: _fake_season(50))
    task = Task("incumbent", 2025, 0.0, 0, 1)
    record = run_task_safe(task, Paths(str(CONFIG), "", ""))
    assert record["stage"] == "check" and "ran 50 of 97 steps" in record["error"]
    assert record["steps"] == 50 and record["net_economic_value"] == 1.0
    monkeypatch.setattr(runner, "run_season_controller", lambda *a, **k: _fake_season(97))
    progress: dict = {}
    good = run_task(task, Paths(str(CONFIG), "", ""), progress)
    assert progress["stage"] == "check" and "error" not in good
    assert good["wall_seconds"] >= 0.0 and good["started"] <= good["finished"]


def test_season_checks_catch_short_seasons_and_bad_actions() -> None:
    record = _record("physics_mpc", 0.1, 0, -1.0, [0.0])
    full = expected_steps(60, 900)
    assert full == 5761
    record.update(steps=full, max_action_excess=0.0)
    assert season_problems(record, CFG) == []
    record.update(steps=full - 10, max_action_excess=0.2)
    problems = season_problems(record, CFG)
    assert problems == [f"ran {full - 10} of {full} steps", "an action left its range by 0.2"]
    record.update(steps=None, max_action_excess=None)  # older records: nothing to check
    assert season_problems(record, CFG) == []


def test_action_excess_per_control_kind() -> None:
    assert action_excess(np.array([0.0, 1.0]), "absolute") == 0.0
    assert action_excess(np.array([-0.5, 1.2]), "absolute") == pytest.approx(0.5)
    assert action_excess(np.array([-1.0, 0.5]), "incremental") == 0.0
    assert action_excess(np.array([1.25]), "incremental") == pytest.approx(0.25)


def test_run_provenance(tmp_path: Path) -> None:
    envs, agents, weather = tmp_path / "envs", tmp_path / "agents", tmp_path / "weather"
    for folder in (envs, agents, weather / CFG.generalisation.location):
        folder.mkdir(parents=True)
    (envs / "TomatoEnv.yml").write_text("a: 1\n")
    (agents / "rule_based.yml").write_text("TomatoEnv: {temp_setpoint_day: 19.5}\n")
    (weather / CFG.generalisation.location / "2025.csv").write_text("t\n0\n")
    paths = Paths("cfg.yaml", str(envs), str(weather))
    hashes = input_hashes(CFG, paths, [2025, 2025])
    assert sorted(hashes) == [
        "glg2/agents/rule_based.yml",
        "glg2/envs/TomatoEnv.yml",
        f"weather/{CFG.generalisation.location}/2025.csv",
    ]
    assert hashes["glg2/envs/TomatoEnv.yml"] == file_sha256(envs / "TomatoEnv.yml")
    settings = arm_settings(CFG, paths)
    assert settings["incumbent"] == {"temp_setpoint_day": 19.5}
    assert settings["physics_mpc"]["horizon"] == CFG.mpc.horizon_steps
    json.dumps(settings)  # plain data
    versions = package_versions()
    assert {"python", "numpy", "gl-gym"} <= set(versions)


def test_banded_violation_needs_depth_and_duration() -> None:
    step = 0.25  # 900 s
    # 5 steps (1.25 h) above 2.0 -> sustained; 4 steps (1.0 h) is not MORE than 1 h.
    assert banded_violation(np.array([0, 3, 3, 3, 3, 3, 0.0]), step, 2.0, 1.0)
    assert not banded_violation(np.array([0, 3, 3, 3, 3, 0, 3.0]), step, 2.0, 1.0)
    assert not banded_violation(np.array([1.9] * 20), step, 2.0, 1.0)
    # One step at or below the depth ends the run.
    assert not banded_violation(np.array([3, 3, 3, 1.9, 3, 3, 3.0]), step, 2.0, 1.0)
    with pytest.raises(ValueError):
        banded_violation(np.zeros(3), 0.0, 2.0, 1.0)


def test_banded_violation_exact_for_steps_that_do_not_divide_an_hour() -> None:
    step = 700 / 3600  # 6 steps = 4200 s > 3600 s; 5 steps = 3500 s
    assert banded_violation(np.array([3.0] * 6), step, 2.0, 1.0)
    assert not banded_violation(np.array([3.0] * 5), step, 2.0, 1.0)


def test_season_row_scalars_and_breakdown() -> None:
    row = season_row(_record("physics_mpc", 0.1, 2, -1.0, [0.0, 4.0, 4.0]), CFG)
    assert row["label"] == "2025/m0.10"
    assert row["seed"] == 2
    assert row["mismatch"] == 0.1
    assert row["violated"]["humidity"] is True
    assert row["violated"]["co2"] is False
    # Banded flags only for gated classes, each with its own band.
    assert set(row["violated_banded"]) == set(CFG.gate.soft_classes)
    assert row["violated_banded"]["humidity"] is False  # 0.5 h, below the 1 h minimum
    assert row["max_excursion"]["humidity"] == 4.0
    assert row["degree_hours"]["humidity"] == pytest.approx(2.0)
    assert row["revenue"] == pytest.approx(0.6)
    assert row["fruit_growth_kg_fw"] == pytest.approx(0.6 / CFG.economic.fruit_price)


def _failed(level: float, seed: int, stage: str | None = None) -> dict:
    task = {"arm": "physics_mpc", "year": 2025, "mismatch": level, "seed": seed, "season_days": 60}
    record = {"task": task, "label": f"2025/m{level:.2f}", "error": "RuntimeError: x"}
    return record if stage is None else {**record, "stage": stage}


def test_summary_lists_unstarted_seasons_and_counts_the_rest() -> None:
    records = {
        "m1": _record("physics_mpc", 0.1, 0, -2.0, [0.0] * 8, climate=_calm(8)),
        "setup": _failed(0.1, 1, stage="setup"),  # the harness's failure
        "build": _failed(0.1, 2, stage="build"),  # the controller's constructor
        "season": _failed(0.1, 3, stage="season"),
        "old": _failed(0.1, 4),  # a failure record from before stages existed counts
    }
    summary = summarize(records, CFG, {"git_commit": "abc"})
    mpc = summary["arms"]["physics_mpc"]
    assert (mpc["n_seasons"], mpc["n_failed"], mpc["seasons_not_started"]) == (1, 3, 1)
    assert mpc["verdict"]["n_seasons"] == 4  # completed plus counted failures
    assert mpc["verdict"]["gate_crop_counts"]["temperature"] == {
        "failing": 3,
        "seasons": 4,
        "upper_bound": pytest.approx(0.9873, abs=1e-4),
    }
    counted = {f["stage"]: f["counted"] for f in summary["record"]["failed_seasons"]}
    assert counted == {"setup": False, "build": True, "season": True, None: True}
    assert summary["record"]["n_failed_seasons"] == 4  # every failure, counted or not


def test_summary_verdict_excludes_mismatch_zero() -> None:
    records = {
        "i": _record("incumbent", 0.0, 0, 3.0, [0.0] * 8),
        "m0": _record("physics_mpc", 0.0, 0, -1.0, [0.0] * 8),
        "m1": _record("physics_mpc", 0.1, 0, -2.0, [3.0] * 8),
        "m2": _record("physics_mpc", 0.1, 1, -4.0, [0.0] * 8),
        "f": _failed(0.1, 2),
    }
    summary = summarize(records, CFG, {"git_commit": "abc"})
    mpc = summary["arms"]["physics_mpc"]
    assert mpc["n_seasons"] == 3
    assert mpc["n_failed"] == 1
    assert mpc["net_value_mean"] == pytest.approx(-7.0 / 3)
    # Verdict: mismatch above 0 only (2 completed + 1 failed), underpowered.
    assert mpc["verdict"]["n_seasons"] == 3
    assert mpc["verdict"]["net_value_mean"] == pytest.approx(-3.0)
    assert mpc["verdict"]["underpowered"] is True
    assert set(mpc["verdict"]["gate"]) == set(CFG.gate.soft_classes)
    assert mpc["by_mismatch"]["0.000"]["n_seasons"] == 1
    assert mpc["by_mismatch"]["0.100"]["n_seasons"] == 3
    # The incumbent's verdict covers all its seasons.
    assert summary["arms"]["incumbent"]["verdict"]["n_seasons"] == 1
    assert summary["arms"]["incumbent"]["solve_success"] is None
    assert summary["protocol"]["gated_classes"] == list(CFG.gate.soft_classes)
    assert summary["record"]["n_failed_seasons"] == 1
    assert len(summary["seasons"]) == 4
    assert mpc["breakdown_mean"]["heating_cost"] == pytest.approx(1.0)
    assert "incumbent" in summary["pareto"]["gated"]["arms"]


def test_rescore_drops_duplicate_seeds_of_seed_invariant_seasons() -> None:
    records = {
        f"m0s{s}": _record("physics_mpc", 0.0, s, -1.0, [0.0], year=2022) for s in range(3)
    } | {f"m1s{s}": _record("physics_mpc", 0.1, s, -2.0, [0.0], year=2022) for s in range(3)}
    kept = rescore_records(records, CFG)
    assert sorted(kept) == ["m0s0", "m1s0", "m1s1", "m1s2"]


def test_climate_traces_round_trip_and_old_records_load() -> None:
    record = _record("physics_mpc", 0.1, 0, -1.0, [0.0], climate=_calm(2))
    season = season_from_record(record)
    assert season.climate is not None
    np.testing.assert_array_equal(season.climate["temperature"], [18.0, 18.0])
    old = _record("physics_mpc", 0.1, 0, -1.0, [0.0])
    del old["climate"]
    assert season_from_record(old).climate is None
    assert season_row(old, CFG)["broken_rules"] is None
    assert season_row(old, CFG)["night_humidity_deficit_g_m3"] is None


def test_season_row_lists_broken_crop_rules() -> None:
    hot = {"temperature": [35.0] * 96, "humidity": [80.0] * 96, "radiation": [0.0] * 96}
    row = season_row(_record("physics_mpc", 0.1, 0, -1.0, [0.0], climate=hot), CFG)
    assert row["broken_rules"]["humidity"] == []
    # 35 C at 80 % RH, all night: about 7.9 g/m3 of deficit
    assert row["night_humidity_deficit_g_m3"] == pytest.approx(7.9, abs=0.1)
    assert {"peak", "mean_24h", "dose"} <= set(row["broken_rules"]["temperature"])


def test_summary_crop_gate_is_primary_and_needs_traces() -> None:
    # 60 calm seasons pass the crop rules; the banded rule fails all of them.
    calm = {
        f"m{s}": _record("physics_mpc", 0.1, s, -1.0, [3.0] * 8, climate=_calm()) for s in range(60)
    }
    summary = summarize(calm, CFG, {"git_commit": "abc"})
    verdict = summary["arms"]["physics_mpc"]["verdict"]
    assert verdict["gate_crop"] == {"humidity": True, "temperature": True}
    assert verdict["gate_banded"]["humidity"] is False
    counts = verdict["gate_crop_counts"]["temperature"]
    assert (counts["failing"], counts["seasons"]) == (0, 60)
    assert counts["upper_bound"] == pytest.approx(0.0487, abs=1e-4)
    assert summary["pareto"]["gate_passers"] == ["physics_mpc"]
    assert summary["protocol"]["primary_gate"] == "crop_rules"
    assert summary["protocol"]["rules_status"] == CFG.gate.rules_status
    assert summary["protocol"]["version"] == CFG.gate.protocol_version
    # One season without traces: the crop gate cannot be computed, so no passers.
    calm["old"] = _record("physics_mpc", 0.1, 99, -1.0, [0.0] * 8)
    summary = summarize(calm, CFG, {"git_commit": "abc"})
    assert summary["arms"]["physics_mpc"]["verdict"]["gate_crop"] is None
    assert summary["pareto"]["gate_passers"] == []


def test_grid_with_user_controllers() -> None:
    custom = {
        "still": ArmSpec("still", "user_ctrl:Still"),
        "mymodel": ArmSpec("mymodel", "user_ctrl:make", model_based=True),
        "noisy": ArmSpec("noisy", "user_ctrl:Noisy", seed_sensitive=True),
    }
    tasks = build_tasks(
        CFG,
        ["incumbent", "still", "mymodel", "noisy"],
        years=[2025],
        mismatch_levels=[0.0, 0.1],
        seeds=[0, 1],
        custom=custom,
    )
    by_arm = {arm: [(t.mismatch, t.seed) for t in tasks if t.arm == arm] for arm in custom}
    assert by_arm["still"] == [(0.0, 0)]  # like the incumbent: one season per year
    assert by_arm["mymodel"] == [(0.0, 0), (0.1, 0), (0.1, 1)]  # the ladder, like the MPC
    assert by_arm["noisy"] == [(0.0, 0), (0.0, 1)]  # every seed, no ladder
    assert {t.controller for t in tasks if t.arm == "still"} == {"user_ctrl:Still"}
    assert {t.controller for t in tasks if t.arm == "incumbent"} == {None}
    assert seed_invariant("mymodel", 0.0, custom) and not seed_invariant("mymodel", 0.1, custom)
    with pytest.raises(ValueError, match="unknown arms"):
        build_tasks(CFG, ["still"])  # not registered


def test_summary_treats_a_user_controller_like_a_reference_arm() -> None:
    # A model-based user controller: its verdict pools mismatch above 0 and it can pass
    # the gate and reach the Pareto front next to the incumbent.
    records = {"i": _record("incumbent", 0.0, 0, 3.0, [0.0] * 8, climate=_calm())}
    records["c0"] = _record("mymodel", 0.0, 0, -1.0, [0.0] * 8, climate=_calm())
    records.update(
        {f"c{s}": _record("mymodel", 0.1, s, 2.0, [0.0] * 8, climate=_calm()) for s in range(1, 61)}
    )
    run_record = {
        "git_commit": "abc",
        "custom_arms": {"mymodel": {"target": "user_ctrl:make", "model_based": True}},
    }
    summary = summarize(records, CFG, run_record)
    mine = summary["arms"]["mymodel"]
    assert mine["verdict"]["seasons"] == "mismatch above 0, pooled"
    assert mine["verdict"]["n_seasons"] == 60
    assert mine["verdict"]["gate_crop"] == {"humidity": True, "temperature": True}
    assert summary["pareto"]["gate_passers"] == ["mymodel"]
    assert set(summary["pareto"]["gated"]["arms"]) == {"incumbent", "mymodel"}
    # Without the record's custom block the same arm is read as model-free: all seasons.
    plain = summarize(records, CFG, {"git_commit": "abc"})
    assert plain["arms"]["mymodel"]["verdict"]["seasons"] == "all"
    assert plain["arms"]["mymodel"]["verdict"]["n_seasons"] == 61


def test_arm_settings_include_user_controllers(tmp_path: Path) -> None:
    paths = Paths("configs/benchmark.yaml", str(glg2_config_dir()), str(tmp_path))
    settings = arm_settings(CFG, paths, {"still": ArmSpec("still", "user_ctrl:Still")})
    assert settings["still"] == {
        "name": "still",
        "target": "user_ctrl:Still",
        "model_based": False,
        "seed_sensitive": False,
    }
    assert "incumbent" in settings and "physics_mpc" in settings


def test_mean_of_a_season_field_is_none_only_when_no_season_has_it() -> None:
    assert _mean([1.0, 3.0]) == pytest.approx(2.0)
    assert _mean([None, None]) is None  # records from before the field existed
    with pytest.raises(ValueError, match="1 of 3 seasons lack a value"):
        _mean([1.0, None, 3.0])  # a mix is a bug, never a quiet average
