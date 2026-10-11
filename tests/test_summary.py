"""Tests for the gated summary: rows, verdicts, gates, rescoring."""

from pathlib import Path

import numpy as np
import pytest
from synthetic import _calm, _failed, _record

from phytotron.config import load_config
from phytotron.metrics.safety import banded_violation
from phytotron.summary import _mean, rescore_records, season_row, summarize

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "benchmark.yaml"
CFG = load_config(CONFIG)


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
    # Without the record's registered block the same arm is read as model-free: all seasons.
    plain = summarize(records, CFG, {"git_commit": "abc"})
    assert plain["arms"]["mymodel"]["verdict"]["seasons"] == "all"
    assert plain["arms"]["mymodel"]["verdict"]["n_seasons"] == 61


def test_mean_of_a_season_field_is_none_only_when_no_season_has_it() -> None:
    assert _mean([1.0, 3.0]) == pytest.approx(2.0)
    assert _mean([None, None]) is None  # records from before the field existed
    with pytest.raises(ValueError, match="1 of 3 seasons lack a value"):
        _mean([1.0, None, 3.0])  # a mix is a bug, never a quiet average
