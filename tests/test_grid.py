"""Tests for the benchmark grid: tasks, their axes, seeds and resume."""

from pathlib import Path

import pytest
from synthetic import _record

from phytotron.config import load_config
from phytotron.controllers.registry import ArmSpec
from phytotron.grid import Task, build_tasks, seed_invariant
from phytotron.records import FailureRecord, pending

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


def test_pending_keeps_failures_unless_retry_is_asked() -> None:
    a, b, c = (Task("physics_mpc", 2025, 0.0, s, 60) for s in range(3))
    done = {
        a.key: _record("physics_mpc", 0.0, 0, 1.0, [0.0]),
        b.key: FailureRecord(b, b.label, "RuntimeError: boom", stage="season"),
    }
    assert pending([a, b, c], done) == [c]
    assert pending([a, b, c], done, retry_failed=True) == [b, c]
    assert pending([a], {}) == [a]


def test_grid_with_user_controllers() -> None:
    registered = {
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
        registered=registered,
    )
    by_arm = {arm: [(t.mismatch, t.seed) for t in tasks if t.arm == arm] for arm in registered}
    assert by_arm["still"] == [(0.0, 0)]  # like the incumbent: one season per year
    assert by_arm["mymodel"] == [(0.0, 0), (0.1, 0), (0.1, 1)]  # the ladder, like the MPC
    assert by_arm["noisy"] == [(0.0, 0), (0.0, 1)]  # every seed, no ladder
    assert {t.controller for t in tasks if t.arm == "still"} == {"user_ctrl:Still"}
    assert {t.controller for t in tasks if t.arm == "incumbent"} == {None}
    assert seed_invariant("mymodel", 0.0, registered) and not seed_invariant(
        "mymodel", 0.1, registered
    )
    with pytest.raises(ValueError, match="unknown arms"):
        build_tasks(CFG, ["still"])  # not registered
