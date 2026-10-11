"""Tests for the config-driven controller settings and simulator overrides."""

from pathlib import Path

import pytest

from phytotron.arms import incumbent_params, mpc_settings
from phytotron.config import load_config
from phytotron.envs.glg2 import scenario_overrides

CFG = load_config(Path(__file__).resolve().parent.parent / "configs" / "benchmark.yaml")


def test_mpc_settings_come_from_config() -> None:
    settings = mpc_settings(CFG)
    assert settings["horizon"] == CFG.mpc.horizon_steps
    assert settings["terminal_weight"] == CFG.mpc.terminal_weight
    assert settings["dt"] == float(CFG.raw["episode"]["timestep_seconds"])
    assert settings["climate_bounds"]["rh_percent"] == tuple(CFG.raw["constraints"]["rh_percent"])
    assert settings["economic"] is CFG.economic


def test_incumbent_params_read_from_rule_based_yml(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "rule_based.yml").write_text("TomatoEnv:\n  tHeatBand: 1.5\n")
    assert incumbent_params(tmp_path / "envs") == {"tHeatBand": 1.5}


def test_simulator_gets_band_and_prices_from_config() -> None:
    over = scenario_overrides(CFG)
    band = CFG.raw["constraints"]
    assert (over["constraints"]["rh_min"], over["constraints"]["rh_max"]) == tuple(
        band["rh_percent"]
    )
    assert over["constraints"]["temp_max"] == band["air_temp_c"][1]
    assert over["reward_params"]["fruit_price"] == CFG.economic.fruit_price
    assert over["reward_params"]["heating_price"] == CFG.economic.heating_price
    assert "pen_weights" not in over["reward_params"]


def test_gate_classes_and_validation() -> None:
    assert CFG.gate.soft_classes == ("humidity", "temperature")
    assert CFG.gate.band("temperature") == (2.0, 1.0)
    with pytest.raises(ValueError, match="no banded rule"):
        CFG.gate.band("co2")
