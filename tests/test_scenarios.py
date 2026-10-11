"""Tests for the shipped scenario files: unchanged from upstream, and in agreement with
configs/benchmark.yaml (the single source of the band and prices)."""

from pathlib import Path

import yaml

from phytotron.arms import incumbent_params
from phytotron.config import load_config
from phytotron.envs.glg2 import scenario_overrides
from phytotron.scenarios import (
    GLG2_RELEASE,
    UPSTREAM,
    check_scenario_files,
    glg2_config_dir,
    scenario_files,
)

CFG = load_config(Path(__file__).resolve().parent.parent / "configs" / "benchmark.yaml")


def test_shipped_files_match_upstream_hashes() -> None:
    assert check_scenario_files() == []
    assert set(scenario_files()) == set(UPSTREAM)
    assert all(p.is_file() for p in scenario_files().values())
    assert CFG.testbed["release"] == GLG2_RELEASE


def test_benchmark_band_and_prices_equal_the_shipped_scenario() -> None:
    spec = yaml.safe_load((glg2_config_dir() / "TomatoEnv.yml").read_text())["TomatoEnv"]
    ours = scenario_overrides(CFG)
    for key, value in ours["constraints"].items():
        assert float(spec["constraints"][key]) == value, key
    for key, value in ours["reward_params"].items():
        assert float(spec["reward_params"][key]) == value, key


def test_incumbent_parameters_load_from_the_shipped_copy() -> None:
    params = incumbent_params(glg2_config_dir())
    assert params["rh_max"] == CFG.raw["constraints"]["rh_percent"][1]
    assert {"temp_setpoint_day", "temp_setpoint_night", "co2_day"} <= set(params)
