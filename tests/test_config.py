"""Tests that the single-source config loads, validates, and self-checks."""

import re
from pathlib import Path

import pytest
import yaml

from phytotron.config import (
    BenchmarkConfig,
    EconomicConfig,
    GateClass,
    GateConfig,
    GeneralisationConfig,
    load_config,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "benchmark.yaml"

_OK_GATE = GateConfig(
    epsilon=0.05, delta=0.05, n_seasons_min=60, method="clopper_pearson_one_sided_upper"
)
_OK_GEN = GeneralisationConfig(
    location="X",
    train_years=(2019, 2020),
    heldout_years=(2021,),
    season_start_day=59,
    season_length_days=60,
)


def test_canonical_config_loads_and_validates() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.gate.epsilon == 0.05
    assert cfg.gate.delta == 0.05
    assert cfg.gate.n_seasons_min >= 59  # enough for a clean pass
    assert cfg.testbed["release"] == "v0.2.1"
    # the pinned weather-year split (most recent 12 complete years)
    assert cfg.generalisation.train_years == (2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021)
    assert cfg.generalisation.heldout_years == (2022, 2023, 2024, 2025)
    assert not set(cfg.generalisation.train_years) & set(cfg.generalisation.heldout_years)
    assert cfg.generalisation.season_start_day == 59
    assert cfg.generalisation.season_length_days == 60


def test_protocol_version_is_loaded_and_checked() -> None:
    cfg = load_config(CONFIG_PATH)
    assert cfg.gate.protocol_version >= 2  # protocol 2: peak rule with a 30-minute minimum
    assert cfg.gate.rules("temperature")["peak"]["min_minutes"] == 30.0
    bad = GateConfig(
        epsilon=0.05,
        delta=0.05,
        n_seasons_min=60,
        method="clopper_pearson_one_sided_upper",
        protocol_version=0,
    )
    with pytest.raises(ValueError, match="protocol_version"):
        bad.validate()


def test_validate_rejects_too_few_seasons() -> None:
    bad = BenchmarkConfig(
        raw={"testbed": {}},
        gate=GateConfig(
            epsilon=0.05, delta=0.05, n_seasons_min=10, method="clopper_pearson_one_sided_upper"
        ),
        generalisation=_OK_GEN,
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_validate_rejects_out_of_range_epsilon() -> None:
    bad = BenchmarkConfig(
        raw={"testbed": {}},
        gate=GateConfig(
            epsilon=1.5, delta=0.05, n_seasons_min=60, method="clopper_pearson_one_sided_upper"
        ),
        generalisation=_OK_GEN,
    )
    with pytest.raises(ValueError):
        bad.validate()


def test_validate_rejects_leaky_split() -> None:
    bad = BenchmarkConfig(
        raw={"testbed": {}},
        gate=_OK_GATE,
        generalisation=GeneralisationConfig(
            location="X",
            train_years=(2019, 2020),
            heldout_years=(2020, 2021),
            season_start_day=59,
            season_length_days=60,
        ),
    )
    with pytest.raises(ValueError, match="leaks"):
        bad.validate()


def test_canonical_config_loads_economic_block() -> None:
    cfg = load_config(CONFIG_PATH)
    assert isinstance(cfg.economic, EconomicConfig)
    assert cfg.economic.penalty_weights == (5.0e-5, 5.0e-3, 7.0e-4)  # [CO2, temp, RH]
    assert cfg.economic.cost_input_indices == (0, 1, 4)  # heating, CO2, lamp


def test_l5_economic_applies_paper_overrides() -> None:
    cfg = load_config(CONFIG_PATH)
    ec = cfg.l5_economic()
    # the paper's fruit price is 20 EUR/kg{DW} = 1.2 EUR/kg{FW} / dmfm 0.06
    assert ec.fruit_price == 1.2
    assert ec.dmfm == 0.06
    assert ec.elec_price == 0.20
    assert ec.price_per_kg_dw == pytest.approx(20.0)
    # unchanged fields fall through from the base economic block
    assert ec.heating_price == cfg.economic.heating_price


def test_validate_rejects_episode_length_that_disagrees_with_season() -> None:
    bad = BenchmarkConfig(
        raw={"episode": {"crop_days": 30, "steps": 2880, "timestep_seconds": 900}},
        gate=_OK_GATE,
        generalisation=_OK_GEN,
    )
    with pytest.raises(ValueError, match="crop_days"):
        bad.validate()


def test_validate_rejects_step_count_that_disagrees_with_season() -> None:
    bad = BenchmarkConfig(
        raw={"episode": {"crop_days": 60, "steps": 5000, "timestep_seconds": 900}},
        gate=_OK_GATE,
        generalisation=_OK_GEN,
    )
    with pytest.raises(ValueError, match="steps"):
        bad.validate()


def test_canonical_config_has_crop_rules_for_every_gated_class() -> None:
    gate = load_config(CONFIG_PATH).gate
    assert gate.primary == "crop_rules"
    assert gate.rules_status in ("provisional", "fixed")
    for name in gate.soft_classes:
        assert gate.rules(name)
    assert list(gate.rules("temperature")) == ["mean_6d", "mean_24h", "dose", "peak"]
    assert list(gate.rules("humidity")) == ["humid_hours", "longest_run"]


def _gate(**changes: object) -> GateConfig:
    soft = GateClass("humidity", "soft", depth=2.0, min_hours=1.0, rules=changes.pop("rules", {}))  # type: ignore[arg-type]
    return GateConfig(
        epsilon=0.05,
        delta=0.05,
        n_seasons_min=60,
        method="clopper_pearson_one_sided_upper",
        classes=(soft,),
        **changes,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("gate", "message"),
    [
        (_gate(), "needs crop rules"),
        (_gate(rules={"r": {"kind": "peak"}}), "rule 'r'"),
        (_gate(rules={"r": {"kind": "peak", "high": 1.0}}, primary="other"), "primary"),
        (_gate(rules={"r": {"kind": "peak", "high": 1.0}}, rules_status="maybe"), "rules_status"),
    ],
)
def test_validate_rejects_bad_crop_rule_setup(gate: GateConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        gate.validate()


def test_banded_primary_needs_no_crop_rules() -> None:
    _gate(primary="banded").validate()


# The top-level blocks of configs/benchmark.yaml. Adding one here is the conscious act the
# config header asks for: say in the YAML who reads it, or that it is documentation only.
TOP_LEVEL_BLOCKS = (
    "testbed",
    "episode",
    "controllers",
    "mpc",
    "objective",
    "constraints",
    "gate",
    "pareto",
    "mismatch",
    "generalisation",
    "checks",
    "reporting",
    "economic",
    "l5_reproduction",
)


def test_config_has_exactly_the_known_blocks() -> None:
    data = yaml.safe_load(CONFIG_PATH.read_text())
    assert tuple(data) == TOP_LEVEL_BLOCKS


def test_every_block_says_who_reads_it() -> None:
    lines = CONFIG_PATH.read_text().splitlines()
    headers = {m.group(1): line for line in lines if (m := re.match(r"^([a-z0-9_]+):", line))}
    assert set(headers) == set(TOP_LEVEL_BLOCKS)
    for name, line in headers.items():
        assert "read by:" in line or "documentation only" in line, name


def test_validate_rejects_a_gate_method_the_code_does_not_implement() -> None:
    gate = GateConfig(epsilon=0.05, delta=0.05, n_seasons_min=60, method="wilson")
    with pytest.raises(ValueError, match=r"gate\.method"):
        gate.validate()
