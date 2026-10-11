"""Loader for the single source of truth, ``configs/benchmark.yaml``.

Every benchmark constant is read from here; nothing in the package hardcodes one.
``BenchmarkConfig.validate`` enforces internal consistency, including that the
declared season count is enough for the chosen ``(epsilon, delta)`` gate.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from phytotron.metrics.crop_rules import check_rule
from phytotron.metrics.gate import min_seasons

_REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"
_PACKAGED_CONFIG = Path(__file__).resolve().parent / "configs" / "benchmark.yaml"  # wheel


def default_config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Where the commands find ``benchmark.yaml`` when ``--config`` is not given.

    ``PHYTOTRON_CONFIG`` if set; else ``configs/benchmark.yaml`` in the current directory
    if it exists; else the one in the source checkout this package is imported from; else
    the copy inside the installed package.
    """
    env = os.environ if environ is None else environ
    if env.get("PHYTOTRON_CONFIG"):
        return Path(env["PHYTOTRON_CONFIG"])
    local = Path.cwd() / "configs" / "benchmark.yaml"
    if local.is_file():
        return local
    return _REPO_CONFIG if _REPO_CONFIG.is_file() else _PACKAGED_CONFIG


GATE_KINDS = ("soft", "reported")
GATE_METHODS = ("clopper_pearson_one_sided_upper",)  # what phytotron.metrics.gate implements
VERDICTS = ("pooled_mismatch_above_zero",)
PRIMARY_GATES = ("crop_rules", "banded")
RULES_STATUSES = ("provisional", "fixed")


@dataclass(frozen=True)
class GateClass:
    """One climate class: gated ("soft") or profiled only ("reported").

    A soft class has crop rules (``rules``, name -> spec, see
    :mod:`phytotron.metrics.crop_rules`) and a banded rule: a season violates the banded
    rule only if the class stays more than ``depth`` outside its band for more than
    ``min_hours`` consecutive hours.
    """

    name: str
    kind: str
    depth: float | None = None
    min_hours: float | None = None
    rules: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def gated(self) -> bool:
        return self.kind == "soft"


@dataclass(frozen=True)
class GateConfig:
    epsilon: float
    delta: float
    n_seasons_min: int
    method: str
    classes: tuple[GateClass, ...] = ()
    verdict: str = "pooled_mismatch_above_zero"
    primary: str = "crop_rules"  # which per-season rule decides the ranking
    rules_status: str = "provisional"  # "fixed" once the horticulturist has set the numbers
    protocol_version: int = 1  # raised with every change to a rule; printed on every result

    @property
    def soft_classes(self) -> tuple[str, ...]:
        """Names of the gated classes, in config order."""
        return tuple(c.name for c in self.classes if c.gated)

    def band(self, name: str) -> tuple[float, float]:
        """``(depth, min_hours)`` of a soft class's banded rule."""
        cls = next(c for c in self.classes if c.name == name)
        if cls.depth is None or cls.min_hours is None:
            raise ValueError(f"class {name!r} has no banded rule")
        return cls.depth, cls.min_hours

    def validate(self) -> None:
        """Raise ``ValueError`` if a class, rule, verdict or gate choice is invalid."""
        for cls in self.classes:
            if cls.kind not in GATE_KINDS:
                raise ValueError(f"gate class {cls.name!r}: kind must be one of {GATE_KINDS}")
            if cls.gated and (cls.depth is None or cls.min_hours is None):
                raise ValueError(f"soft gate class {cls.name!r} needs banded depth and min_hours")
            if cls.gated and self.primary == "crop_rules" and not cls.rules:
                raise ValueError(f"soft gate class {cls.name!r} needs crop rules")
            for rule, spec in cls.rules.items():
                try:
                    check_rule(spec)
                except ValueError as err:
                    raise ValueError(f"class {cls.name!r}, rule {rule!r}: {err}") from err
        if self.primary not in PRIMARY_GATES:
            raise ValueError(f"gate.primary must be one of {PRIMARY_GATES}")
        if self.rules_status not in RULES_STATUSES:
            raise ValueError(f"gate.rules_status must be one of {RULES_STATUSES}")
        if self.verdict not in VERDICTS:
            raise ValueError(f"gate.verdict must be one of {VERDICTS}, got {self.verdict!r}")
        if self.method not in GATE_METHODS:
            raise ValueError(f"gate.method must be one of {GATE_METHODS}, got {self.method!r}")
        if self.protocol_version < 1:
            raise ValueError("gate.protocol_version must be 1 or higher")

    def rules(self, name: str) -> dict[str, dict[str, Any]]:
        """The crop rules of a soft class, in config order."""
        return next(c for c in self.classes if c.name == name).rules


@dataclass(frozen=True)
class MpcConfig:
    """The physics-only MPC arm's setup."""

    horizon_steps: int
    terminal_weight: float


@dataclass(frozen=True)
class GeneralisationConfig:
    """The season setup and the weather-year split: train on some years, hold out others."""

    location: str
    train_years: tuple[int, ...]
    heldout_years: tuple[int, ...]
    season_start_day: int  # day of year the season starts
    season_length_days: int


@dataclass(frozen=True)
class EconomicConfig:
    """Economic-MPC objective constants (the GLG2 ``GreenhouseReward`` EPI form).

    Prices, the [CO2, temp, RH] penalty weights, and the GreenLight ``p[]`` / ``x[]``
    indices the cost and gain terms use. Injected into the MPC; never hardcoded.
    """

    fruit_price: float  # EUR/kg fresh weight
    dmfm: float  # dry-matter / fresh-matter ratio
    heating_price: float  # EUR/kWh
    co2_price: float  # EUR/kg
    elec_price: float  # EUR/kWh
    penalty_weights: tuple[float, float, float]  # order [CO2, temp, RH]
    param_indices: dict[str, int]  # GreenLight p[] indices for cost conversions
    fruit_state_index: int  # x[] index of cFruit dry matter
    cost_input_indices: tuple[int, int, int]  # u[] indices: heating, CO2, lamp

    @property
    def price_per_kg_dw(self) -> float:
        """Effective fruit price per kg dry weight (``fruit_price / dmfm``)."""
        return self.fruit_price / self.dmfm


def _economic_from(block: dict[str, Any]) -> EconomicConfig:
    return EconomicConfig(
        fruit_price=float(block["fruit_price_eur_per_kg_fw"]),
        dmfm=float(block["dmfm"]),
        heating_price=float(block["heating_price_eur_per_kwh"]),
        co2_price=float(block["co2_price_eur_per_kg"]),
        elec_price=float(block["elec_price_eur_per_kwh"]),
        penalty_weights=tuple(float(w) for w in block["penalty_weights"]),  # type: ignore[arg-type]
        param_indices={str(k): int(v) for k, v in block["param_indices"].items()},
        fruit_state_index=int(block["fruit_state_index"]),
        cost_input_indices=tuple(int(i) for i in block["cost_input_indices"]),  # type: ignore[arg-type]
    )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parsed benchmark constants. ``raw`` keeps the untyped rest of the YAML."""

    raw: dict[str, Any]
    gate: GateConfig
    generalisation: GeneralisationConfig
    economic: EconomicConfig | None = None
    mpc: MpcConfig | None = None

    @property
    def testbed(self) -> dict[str, Any]:
        return dict(self.raw["testbed"])

    def l5_economic(self) -> EconomicConfig:
        """Economic config for the L5 reproduction: ``economic`` with the paper overrides."""
        base = dict(self.raw["economic"])
        l5 = self.raw["l5_reproduction"]
        for key in ("fruit_price_eur_per_kg_fw", "dmfm", "elec_price_eur_per_kwh"):
            if key in l5:
                base[key] = l5[key]
        return _economic_from(base)

    def validate(self) -> None:
        """Raise ``ValueError`` if any constant is out of range or inconsistent."""
        if not 0.0 < self.gate.epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0, 1), got {self.gate.epsilon}")
        self.gate.validate()
        if not 0.0 < self.gate.delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {self.gate.delta}")
        required = min_seasons(self.gate.epsilon, self.gate.delta)
        if self.gate.n_seasons_min < required:
            raise ValueError(
                f"n_seasons_min={self.gate.n_seasons_min} is below the {required} "
                f"seasons required for a clean pass at eps={self.gate.epsilon}, "
                f"delta={self.gate.delta}"
            )
        train = set(self.generalisation.train_years)
        heldout = set(self.generalisation.heldout_years)
        if not train or not heldout:
            raise ValueError("train and held-out weather-year lists must be non-empty")
        overlap = train & heldout
        if overlap:
            raise ValueError(
                f"weather-year split leaks: {sorted(overlap)} in both train and held-out"
            )
        episode = self.raw.get("episode")
        if episode is not None:
            days = self.generalisation.season_length_days
            if int(episode["crop_days"]) != days:
                raise ValueError(
                    f"episode.crop_days={episode['crop_days']} disagrees with "
                    f"generalisation.season_length_days={days}"
                )
            steps = days * 86400 // int(episode["timestep_seconds"])
            if int(episode["steps"]) != steps:
                raise ValueError(f"episode.steps={episode['steps']} should be {steps}")


def _gate_class(name: str, spec: dict[str, Any]) -> GateClass:
    banded = spec.get("banded") or {}
    return GateClass(
        name=str(name),
        kind=str(spec["kind"]),
        depth=float(banded["depth"]) if "depth" in banded else None,
        min_hours=float(banded["min_hours"]) if "min_hours" in banded else None,
        rules={str(k): dict(v) for k, v in (spec.get("rules") or {}).items()},
    )


def load_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate ``benchmark.yaml``."""
    data = yaml.safe_load(Path(path).read_text())
    gate_raw = data["gate"]
    gate = GateConfig(
        epsilon=float(gate_raw["epsilon"]),
        delta=float(gate_raw["delta"]),
        n_seasons_min=int(gate_raw["n_seasons_min"]),
        method=str(gate_raw["method"]),
        classes=tuple(_gate_class(name, spec) for name, spec in gate_raw["classes"].items()),
        verdict=str(gate_raw["verdict"]),
        primary=str(gate_raw.get("primary", "crop_rules")),
        rules_status=str(gate_raw.get("rules_status", "provisional")),
        protocol_version=int(gate_raw.get("protocol_version", 1)),
    )
    gen_raw = data["generalisation"]
    generalisation = GeneralisationConfig(
        location=str(gen_raw["location"]),
        train_years=tuple(int(y) for y in gen_raw["weather_years_train"]),
        heldout_years=tuple(int(y) for y in gen_raw["weather_years_heldout"]),
        season_start_day=int(gen_raw["season_start_day"]),
        season_length_days=int(gen_raw["season_length_days"]),
    )
    economic = _economic_from(data["economic"]) if "economic" in data else None
    mpc = (
        MpcConfig(
            horizon_steps=int(data["mpc"]["horizon_steps"]),
            terminal_weight=float(data["mpc"]["terminal_weight"]),
        )
        if "mpc" in data
        else None
    )
    config = BenchmarkConfig(
        raw=data, gate=gate, generalisation=generalisation, economic=economic, mpc=mpc
    )
    config.validate()
    return config
