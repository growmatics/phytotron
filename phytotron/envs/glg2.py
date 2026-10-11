"""GreenLight-Gym2 TomatoEnv adapter.

``season_result_from_infos`` is pure (no simulator) and unit-tested; it parses
the TomatoEnv ``info`` dict into the benchmark's objective (net economic value)
and per-class exceedance series. ``make_tomato_env`` and ``run_season_controller`` require
``gl_gym`` and are integration-only, so ``gl_gym`` is imported lazily.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phytotron.config import BenchmarkConfig

# Constraint class -> the TomatoEnv info field carrying its per-step violation.
VIOLATION_FIELDS: dict[str, str] = {
    "temperature": "temp_violation",
    "co2": "co2_violation",
    "humidity": "rh_violation",
    "lamp": "lamp_violation",
}

# Per-step cost terms in GreenLight-Gym2's step output; they sum to "variable_costs".
# A season runs N + 1 control steps: TomatoEnv ends it when ``timestep >= N`` is seen
# before the step counter moves on (gl_gym/environments/tomato_env.py, line 84).
EXTRA_STEPS = 1
# Range of one action per control kind: absolute controls are set directly (and
# ``step_raw_control`` does not clip them); incremental actions are rates in [-1, 1].
ACTION_RANGES: dict[str, tuple[float, float]] = {
    "absolute": (0.0, 1.0),
    "incremental": (-1.0, 1.0),
}


def expected_steps(season_days: int, timestep_seconds: int) -> int:
    """Control steps in a full season of ``season_days``."""
    return season_days * 86400 // timestep_seconds + EXTRA_STEPS


def action_excess(action: NDArray[np.float64], kind: str) -> float:
    """How far an action leaves its kind's range (0 if inside)."""
    low, high = ACTION_RANGES[kind]
    a = np.asarray(action, dtype=float)
    return float(max(np.max(low - a, initial=0.0), np.max(a - high, initial=0.0), 0.0))


COST_FIELDS: dict[str, str] = {
    "heating_cost": "heat_cost",
    "co2_cost": "co2_cost",
    "electricity_cost": "elec_cost",
}


@dataclass(frozen=True)
class SeasonResult:
    """One season's benchmark outputs."""

    net_economic_value: float  # revenue minus VARIABLE (energy + resource) cost over the season
    exceedances: dict[str, NDArray[np.float64]]  # per class, per-step magnitude (>= 0)
    online_cost_seconds: float = 0.0  # mean controller decision time per step (Pareto axis)
    # Fraction of NLP solves that converged or salvaged a usable iterate (vs held the last
    # control): a deployment-reliability metric for the MPC arms; None for solver-free arms.
    solve_success: float | None = None
    # Where the net value comes from, EUR/m2 over the season (None in records made before
    # the breakdown existed). net = revenue - heating - co2 - electricity.
    revenue: float | None = None
    heating_cost: float | None = None
    co2_cost: float | None = None
    electricity_cost: float | None = None
    steps: int | None = None  # control steps actually run
    # Raw climate per step, after each step: "temperature" (air, C), "humidity" (air,
    # %RH), and "radiation" (outdoor global radiation during the step, W/m2). The crop
    # rules of the gate read these (None in records made before them).
    climate: dict[str, NDArray[np.float64]] | None = None
    # MPC solve tally per outcome ("clean", "salvage", "hold"); None for solver-free arms.
    solve_counts: dict[str, int] | None = None
    # Largest distance of any action outside its control kind's range, over the season.
    max_action_excess: float | None = None


def season_result_from_infos(infos: Sequence[dict[str, Any]]) -> SeasonResult:
    """Parse per-step TomatoEnv info dicts into a ``SeasonResult`` (pure).

    Net economic value is revenue minus the VARIABLE (energy and resource) costs only,
    per the benchmark objective (``configs/benchmark.yaml``) and GLG2's own
    ``GreenhouseReward.profit`` (which likewise excludes fixed costs). Fixed costs are a
    control-independent constant offset, so they do not affect ranking or the Pareto
    front; excluding them makes the number the true operating profit.
    """
    if len(infos) == 0:
        raise ValueError("need at least one step of info")
    # Validate the expected keys are present and REFUSE to default them: a missing key
    # (e.g. a gl_gym version renamed it) must fail loudly, not silently report 0 revenue /
    # 0 violations -- that would be a plausible-but-wrong number that nobody notices.
    required = (
        "revenue",
        "variable_costs",
        *COST_FIELDS.values(),
        *VIOLATION_FIELDS.values(),
    )
    missing = [k for k in required if k not in infos[0]]
    if missing:
        raise KeyError(
            f"TomatoEnv info dict is missing expected keys {missing}; the installed gl_gym "
            "version may have changed. Refusing to default them to 0 (would report "
            "plausible-but-wrong metrics)."
        )
    revenue = np.array([float(i["revenue"]) for i in infos])
    variable = np.array([float(i["variable_costs"]) for i in infos])
    net = float(revenue.sum() - variable.sum())
    exceedances = {
        cls: np.array([abs(float(i[field])) for i in infos])
        for cls, field in VIOLATION_FIELDS.items()
    }
    costs = {name: float(sum(float(i[key]) for i in infos)) for name, key in COST_FIELDS.items()}
    return SeasonResult(
        net_economic_value=net,
        exceedances=exceedances,
        revenue=float(revenue.sum()),
        heating_cost=costs["heating_cost"],
        co2_cost=costs["co2_cost"],
        electricity_cost=costs["electricity_cost"],
        steps=len(infos),
    )


def default_glg2_paths() -> tuple[Path, Path]:  # pragma: no cover - needs gl_gym
    """Config and weather directories from the installed ``gl_gym`` package."""
    import gl_gym

    root = Path(gl_gym.__file__).resolve().parent
    return root / "configs" / "envs", root / "environments" / "weather"


def paths_from_env(environ: Mapping[str, str]) -> tuple[Path, Path]:
    """The simulator's config directory and the weather directory, checked.

    ``GLG2_CONFIG_DIR`` overrides the scenario files Phytotron ships
    (:mod:`phytotron.scenarios`); ``GLG2_WEATHER_DIR`` defaults to ``weather``. Raises
    ``SystemExit`` with a readable message instead of a traceback when either is missing.
    """
    from phytotron.scenarios import glg2_config_dir

    raw = environ.get("GLG2_CONFIG_DIR")
    config_dir = Path(raw) if raw else glg2_config_dir()
    if not (config_dir / "TomatoEnv.yml").is_file():
        raise SystemExit(
            f"No TomatoEnv.yml in {config_dir}. Unset GLG2_CONFIG_DIR to use the scenario "
            "files Phytotron ships, or point it at <GreenLight-Gym2 checkout>/gl_gym/configs/envs."
        )
    weather_dir = Path(environ.get("GLG2_WEATHER_DIR") or "weather")
    if not weather_dir.is_dir():
        raise SystemExit(
            f"Weather directory {weather_dir} not found. Fetch it with "
            "examples/fetch_weather.py, or set GLG2_WEATHER_DIR (see docs/quickstart.md)."
        )
    return config_dir, weather_dir


def make_tomato_env(
    config_dir: str | Path | None = None,
    weather_dir: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    spec_overrides: dict[str, Any] | None = None,
) -> Any:  # pragma: no cover - needs gl_gym + config/weather data
    """Build a ``TomatoEnv``. Requires ``gl_gym`` installed.

    ``overrides`` patch the base env params (e.g. ``training``, ``start_train_year``);
    ``spec_overrides`` patch the TomatoEnv spec (e.g. ``eval_options``). Defaults to
    the installed package's config and weather directories.
    """
    from gl_gym.common.utils import load_env_params
    from gl_gym.environments.tomato_env import TomatoEnv

    default_config, default_weather = default_glg2_paths()
    config_dir = Path(config_dir) if config_dir is not None else default_config
    weather_dir = Path(weather_dir) if weather_dir is not None else default_weather

    base_params, env_params = load_env_params("TomatoEnv", str(config_dir))
    base_params["weather_data_dir"] = str(weather_dir)
    if overrides:
        base_params.update(overrides)
    for key, value in (spec_overrides or {}).items():
        # merge one level deep, so overriding some prices keeps the other reward params
        if isinstance(value, dict) and isinstance(env_params.get(key), dict):
            env_params[key] = {**env_params[key], **value}
        else:
            env_params[key] = value
    return TomatoEnv(base_env_params=base_params, **env_params)


def scenario_overrides(cfg: BenchmarkConfig) -> dict[str, Any]:
    """The simulator settings that ``configs/benchmark.yaml`` owns: band and prices.

    Passed to the simulator so that scoring (the violation and cost terms GreenLight-Gym2
    computes) and the MPC's objective read the same numbers from one file. The penalty
    weights are not passed: they shape only GreenLight-Gym2's own reward, which Phytotron
    does not score.
    """
    band = cfg.constraints
    if cfg.economic is None:
        raise ValueError("configs/benchmark.yaml has no economic block")
    ec = cfg.economic
    return {
        "constraints": {
            "co2_min": float(band.co2_ppm[0]),
            "co2_max": float(band.co2_ppm[1]),
            "temp_min": float(band.air_temp_c[0]),
            "temp_max": float(band.air_temp_c[1]),
            "rh_min": float(band.rh_percent[0]),
            "rh_max": float(band.rh_percent[1]),
        },
        "reward_params": {
            "fruit_price": ec.fruit_price,
            "dmfm": ec.dmfm,
            "heating_price": ec.heating_price,
            "co2_price": ec.co2_price,
            "elec_price": ec.elec_price,
        },
    }


def make_benchmark_env(
    cfg: BenchmarkConfig,
    year: int,
    *,
    config_dir: str | Path | None = None,
    weather_dir: str | Path | None = None,
    season_days: int | None = None,
) -> Any:  # pragma: no cover - needs gl_gym + config/weather data
    """A ``TomatoEnv`` for one benchmark season: the config's location, start day, length,
    band and prices, and the given weather ``year``."""
    gen = cfg.generalisation
    return make_tomato_env(
        config_dir,
        weather_dir,
        overrides={
            "location": gen.location,
            "start_train_year": year,
            "end_train_year": year,
            "start_train_day": gen.season_start_day,
            "end_train_day": gen.season_start_day,
            "season_length": season_days or gen.season_length_days,
        },
        spec_overrides=scenario_overrides(cfg),
    )


def climate_point(state: NDArray[np.float64]) -> tuple[float, float]:  # pragma: no cover - gl_gym
    """Air temperature (C) and relative humidity (%RH) from a TomatoEnv state vector.

    State 2 is air temperature and 15 the air vapour pressure; the RH conversion is
    GreenLight-Gym2's own, the one its observations and humidity violation use.
    """
    from gl_gym.environments.utils import vaporPres2rh

    temp = float(state[2])
    return temp, float(vaporPres2rh(temp, float(state[15])))


def _with_climate(
    result: SeasonResult, points: list[tuple[float, float]], radiation: list[float]
) -> SeasonResult:
    from dataclasses import replace

    trace = np.asarray(points, dtype=float).reshape(-1, 2)
    climate = {
        "temperature": trace[:, 0],
        "humidity": trace[:, 1],
        "radiation": np.asarray(radiation, dtype=float),
    }
    return replace(result, climate=climate)


def _outdoor_radiation(env: Any) -> float:  # pragma: no cover - needs a live env
    """Outdoor global radiation (W/m2) in the weather row of the coming step (column 0)."""
    return float(env.weather_data[int(env.timestep)][0])


def run_season_controller(
    env: Any,
    controller: Any,
    seed: int,
    forecast_steps: int = 24,
) -> SeasonResult:  # pragma: no cover - needs a live env
    """Run one full season driving a benchmark ``Controller`` via ``act(ControlContext)``.

    This is the uniform harness every arm runs through: each step it assembles a
    ``ControlContext`` (observation, full state, weather forecast, timestep, live env)
    and calls ``controller.act``. ``forecast_steps`` is the disturbance window put in the
    context (>= the largest controller horizon; the env carries a prediction buffer past
    the season end, so the slice is always full).
    """
    from phytotron.controllers.base import ControlContext

    # Controllers output one of two control conventions, and the env has a step for
    # each: "absolute" controls (physics MPC, rule-based) go through step_raw_control,
    # which sets u directly; "incremental" rate actions in [-1,1] (RL) go through the
    # gym step, which does u <- clip(u + a*delta_u_max). Using the wrong one silently
    # feeds the plant different controls than the controller planned.
    kind = getattr(controller, "control_kind", "absolute")
    incremental = kind == "incremental"
    step = env.step if incremental else env.step_raw_control

    import time
    from dataclasses import replace

    obs, _info = env.reset(seed=seed)
    controller.reset(seed)
    infos: list[dict[str, Any]] = []
    points: list[tuple[float, float]] = []
    radiation: list[float] = []
    act_seconds = 0.0
    excess = 0.0
    done = False
    while not done:
        t = int(env.timestep)
        ctx = ControlContext(
            obs=np.asarray(obs, dtype=float),
            state=np.asarray(env.x, dtype=float).flatten(),
            forecast=np.asarray(env.weather_data[t : t + forecast_steps], dtype=float),
            timestep=t,
            env=env,
        )
        a = time.perf_counter()
        action = controller.act(ctx)  # timed: this is the arm's online (deployment) cost
        act_seconds += time.perf_counter() - a
        excess = max(excess, action_excess(action, kind))
        radiation.append(float(ctx.forecast[0][0]))
        obs, _reward, terminated, truncated, info = step(action)
        infos.append(info)
        points.append(climate_point(np.asarray(env.x, dtype=float).flatten()))
        done = bool(terminated or truncated)
    result = _with_climate(season_result_from_infos(infos), points, radiation)
    # Pull the MPC solve-reliability tally if this arm has one (arm 3 directly, arm 4 via
    # its inner ``.mpc``); solver-free arms (rule-based, RL) leave it None.
    stats = getattr(controller, "solve_stats", None) or getattr(
        getattr(controller, "mpc", None), "solve_stats", None
    )
    solve_success = (
        (stats["clean"] + stats["salvage"]) / total
        if stats is not None and (total := sum(stats.values())) > 0
        else None
    )
    return replace(
        result,
        online_cost_seconds=act_seconds / max(len(infos), 1),
        solve_success=solve_success,
        solve_counts=None if stats is None else {k: int(v) for k, v in stats.items()},
        max_action_excess=excess,
    )
