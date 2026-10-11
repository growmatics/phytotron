"""Tests for the pure TomatoEnv info parser (no simulator needed)."""

import numpy as np
import pytest

from phytotron.envs.glg2 import VIOLATION_FIELDS, paths_from_env, season_result_from_infos
from phytotron.metrics.safety import safety_profile_from_exceedance
from phytotron.scenarios import glg2_config_dir


def _info(revenue: float, variable: float, fixed: float, **violations: float) -> dict:
    # split the variable cost 50/30/20 over heating, CO2 and electricity
    base = {
        "revenue": revenue,
        "variable_costs": variable,
        "fixed_costs": fixed,
        "heat_cost": 0.5 * variable,
        "co2_cost": 0.3 * variable,
        "elec_cost": 0.2 * variable,
        "temp_violation": 0.0,
        "co2_violation": 0.0,
        "rh_violation": 0.0,
        "lamp_violation": 0.0,
    }
    base.update(violations)
    return base


def test_net_economic_value_is_revenue_minus_variable_costs() -> None:
    # net excludes fixed costs (a control-independent offset), per the benchmark
    # objective and GLG2's own GreenhouseReward.profit.
    infos = [
        _info(10.0, 2.0, 1.0),
        _info(5.0, 1.0, 1.0),
    ]
    result = season_result_from_infos(infos)
    assert result.net_economic_value == pytest.approx(10.0 + 5.0 - 2.0 - 1.0)


def test_cost_breakdown_and_step_count() -> None:
    result = season_result_from_infos([_info(10.0, 2.0, 1.0), _info(5.0, 1.0, 1.0)])
    assert result.revenue == pytest.approx(15.0)
    assert result.heating_cost == pytest.approx(1.5)
    assert result.co2_cost == pytest.approx(0.9)
    assert result.electricity_cost == pytest.approx(0.6)
    assert result.steps == 2
    costs = result.heating_cost + result.co2_cost + result.electricity_cost
    assert result.revenue - costs == pytest.approx(result.net_economic_value)


def test_exceedances_parsed_per_class() -> None:
    infos = [
        _info(0.0, 0.0, 0.0, temp_violation=1.5, rh_violation=0.0),
        _info(0.0, 0.0, 0.0, temp_violation=0.0, rh_violation=2.0),
    ]
    result = season_result_from_infos(infos)
    assert set(result.exceedances) == set(VIOLATION_FIELDS)
    assert np.allclose(result.exceedances["temperature"], [1.5, 0.0])
    assert np.allclose(result.exceedances["humidity"], [0.0, 2.0])
    # feed one class straight into the safety profile
    prof = safety_profile_from_exceedance([result.exceedances["temperature"]], step_hours=0.25)
    assert prof.incidence == 1.0
    assert prof.worst_excursion == 1.5


def test_empty_infos_rejected() -> None:
    with pytest.raises(ValueError):
        season_result_from_infos([])


def test_missing_info_key_raises_not_defaults() -> None:
    # a dropped/renamed gl_gym key must fail loudly, not silently report 0 (would be a
    # plausible-but-wrong metric)
    incomplete = {
        "variable_costs": 1.0,
        "heat_cost": 1.0,
        "co2_cost": 0.0,
        "elec_cost": 0.0,
        "temp_violation": 0.0,
        "co2_violation": 0.0,
        "rh_violation": 0.0,
        "lamp_violation": 0.0,
    }  # no "revenue"
    with pytest.raises(KeyError, match="missing expected keys"):
        season_result_from_infos([incomplete])


def test_paths_from_env_checks_both_directories(tmp_path) -> None:
    config = tmp_path / "envs"
    config.mkdir()
    weather = tmp_path / "weather"
    with pytest.raises(SystemExit, match=r"No TomatoEnv\.yml"):
        paths_from_env({"GLG2_CONFIG_DIR": str(config), "GLG2_WEATHER_DIR": str(weather)})
    (config / "TomatoEnv.yml").write_text("")
    with pytest.raises(SystemExit, match="Weather directory"):
        paths_from_env({"GLG2_CONFIG_DIR": str(config), "GLG2_WEATHER_DIR": str(weather)})
    weather.mkdir()
    env = {"GLG2_CONFIG_DIR": str(config), "GLG2_WEATHER_DIR": str(weather)}
    assert paths_from_env(env) == (config, weather)
    # Without GLG2_CONFIG_DIR, the shipped scenario files are used.
    shipped, _ = paths_from_env({"GLG2_WEATHER_DIR": str(weather)})
    assert (shipped / "TomatoEnv.yml").is_file()
    assert shipped == glg2_config_dir()
