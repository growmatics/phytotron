"""Tests for user controllers registered by name (``phytotron/controllers/registry.py``)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from phytotron.controllers.registry import (
    ArmSpec,
    build_custom,
    load_target,
    make_factory,
    parse_spec,
    parse_specs,
    record_specs,
    specs_from_record,
    target_source,
)

MODULE = '''
import numpy as np

class Still:
    """Does nothing, every step."""
    control_kind = "absolute"
    def reset(self, seed): pass
    def act(self, ctx): return np.zeros(6)

class NeedsArgs:
    def __init__(self, gain): self.gain = gain
    def reset(self, seed): pass
    def act(self, ctx): return np.zeros(6)

def make(env, cfg, mismatch, seed):
    c = NeedsArgs(gain=mismatch + seed)
    c.env, c.cfg = env, cfg
    return c

class NoAct:
    def reset(self, seed): pass

NOT_A_CONTROLLER = 3
'''


@pytest.fixture
def module_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "user_ctrl.py").write_text(MODULE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(__import__("sys").modules, "user_ctrl", raising=False)
    return tmp_path


def test_parse_spec() -> None:
    assert parse_spec("thermo=my_mod:Thermostat") == ("thermo", "my_mod:Thermostat")
    assert parse_spec(" a_1=pkg.sub.mod:make ") == ("a_1", "pkg.sub.mod:make")
    for bad in ("thermo", "thermo=my_mod", "=m:C", "1x=m:C", "a=m:", "a=m:C:D", "a b=m:C"):
        with pytest.raises(ValueError, match="name=module:Class"):
            parse_spec(bad)


def test_parse_specs_builds_arm_specs_and_checks_names() -> None:
    arms = parse_specs(
        ["a=m:A", "b=m:B", "c=m:C"],
        model_based=["b"],
        seed_sensitive=["c"],
        reserved=("incumbent",),
    )
    assert arms["a"] == ArmSpec("a", "m:A")
    assert arms["b"] == ArmSpec("b", "m:B", model_based=True)
    assert arms["c"] == ArmSpec("c", "m:C", seed_sensitive=True)
    assert parse_specs(None) == {}
    with pytest.raises(ValueError, match="built-in"):
        parse_specs(["incumbent=m:A"], reserved=("incumbent",))
    with pytest.raises(ValueError, match="twice"):
        parse_specs(["a=m:A", "a=m:B"])
    with pytest.raises(ValueError, match="--model-based"):
        parse_specs(["a=m:A"], model_based=["zzz"])


def test_seed_invariance_of_custom_arms() -> None:
    plain = ArmSpec("a", "m:A")
    assert plain.seed_invariant(0.0) and plain.seed_invariant(0.1)
    model = ArmSpec("b", "m:B", model_based=True)
    assert model.seed_invariant(0.0) and not model.seed_invariant(0.1)
    noisy = ArmSpec("c", "m:C", seed_sensitive=True)
    assert not noisy.seed_invariant(0.0)


def test_load_target_imports_from_the_current_directory(module_dir: Path) -> None:
    assert load_target("user_ctrl:Still").__name__ == "Still"
    with pytest.raises(ValueError, match="cannot import"):
        load_target("no_such_module:X")
    with pytest.raises(ValueError, match="no attribute"):
        load_target("user_ctrl:Missing")


def test_make_factory_wraps_a_class_and_passes_a_function(module_dir: Path) -> None:
    still = make_factory(load_target("user_ctrl:Still"))("env", "cfg", 0.1, 3)
    assert type(still).__name__ == "Still"
    made = make_factory(load_target("user_ctrl:make"))("env", "cfg", 0.1, 3)
    assert (made.gain, made.env, made.cfg) == (3.1, "env", "cfg")
    with pytest.raises(ValueError, match="class or a factory"):
        make_factory(3)


def test_build_custom_checks_the_interface(module_dir: Path) -> None:
    controller = build_custom("user_ctrl:make", "env", "cfg", mismatch=0.0, seed=1)
    assert controller.gain == 1.0
    with pytest.raises(ValueError, match="no act"):
        build_custom("user_ctrl:NoAct", "env", "cfg", mismatch=0.0, seed=0)
    with pytest.raises(TypeError):  # a class that needs arguments must come as a factory
        build_custom("user_ctrl:NeedsArgs", "env", "cfg", mismatch=0.0, seed=0)


def test_target_source_names_the_file_and_its_hash(module_dir: Path) -> None:
    source = target_source("user_ctrl:Still")
    assert source["file"] == "user_ctrl.py"  # relative to the current directory
    assert (
        source["sha256"] == hashlib.sha256((module_dir / "user_ctrl.py").read_bytes()).hexdigest()
    )


def test_record_specs_round_trip(module_dir: Path) -> None:
    arms = {"still": ArmSpec("still", "user_ctrl:Still", model_based=True)}
    data = record_specs(arms)
    assert data["still"]["target"] == "user_ctrl:Still"
    assert data["still"]["model_based"] is True
    assert len(data["still"]["sha256"]) == len(hashlib.sha256(b"").hexdigest())
    assert specs_from_record({"custom_arms": data}) == arms
    assert specs_from_record({}) == {}
