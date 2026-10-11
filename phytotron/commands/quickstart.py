"""Quickstart: run one controller through one simulated season and print its scores.

Runs the rule-based incumbent (or the economic MPC) on the reference scenario: a Venlo
tomato greenhouse in Amsterdam weather, season start and length from
``configs/benchmark.yaml``. Prints the net economic value, where it comes from, and per
climate class how often and how far the controller left the allowed band. See
``docs/quickstart.md``.

    GLG2_WEATHER_DIR=weather phytotron quickstart --controller incumbent

The incumbent runs the full season in seconds. The MPC defaults to 3 days (about a
minute); ``--days 60`` runs the full season in about 20 minutes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from phytotron.arms import build_controller
from phytotron.config import default_config_path, load_config
from phytotron.envs.glg2 import (
    make_benchmark_env,
    paths_from_env,
    run_season_controller,
)
from phytotron.metrics.safety import safety_profile_from_exceedance

_ARMS = {"incumbent": "incumbent", "mpc": "physics_mpc"}
_MPC_DEFAULT_DAYS = 3  # a quick look; the full season takes about 20 minutes
# Units per class; "lamp" counts lamp use outside the allowed hours.
_UNITS = {"temperature": "C", "co2": "ppm", "humidity": "%RH", "lamp": "on"}


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """The ``quickstart`` command's arguments, on ``parser`` or on a new one."""
    parser = parser or argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--controller", choices=tuple(_ARMS), default="incumbent")
    parser.add_argument(
        "--config", type=Path, default=default_config_path(), help="configs/benchmark.yaml"
    )
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--days", type=int, default=None, help="default: full season (MPC: 3)")
    parser.add_argument("--mismatch", type=float, default=0.0, help="MPC model error, e.g. 0.1")
    return parser


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - needs a live env
    run(build_parser().parse_args(argv))


def run(args: argparse.Namespace) -> None:  # pragma: no cover - needs a live env
    cfg = load_config(args.config)
    gen = cfg.generalisation
    default_days = _MPC_DEFAULT_DAYS if args.controller == "mpc" else gen.season_length_days
    days = args.days or default_days
    config_dir, weather_dir = paths_from_env(os.environ)
    env = make_benchmark_env(
        cfg, args.year, config_dir=config_dir, weather_dir=weather_dir, season_days=days
    )
    env.reset(seed=0)
    controller = build_controller(
        _ARMS[args.controller], env, cfg, glg2_config_dir=config_dir, mismatch=args.mismatch
    )
    result = run_season_controller(env, controller, seed=0)

    step_hours = float(cfg.raw["episode"]["timestep_seconds"]) / 3600.0
    assert cfg.economic is not None and result.revenue is not None
    print(f"{args.controller}, simulated season, {gen.location} {args.year}, {days} days")
    print(f"  net economic value: {result.net_economic_value:+.3f} EUR/m2")
    print(
        f"    = revenue {result.revenue:.3f} - heating {result.heating_cost:.3f}"
        f" - CO2 {result.co2_cost:.3f} - electricity {result.electricity_cost:.3f}"
    )
    print(f"  fruit growth:       {result.revenue / cfg.economic.fruit_price:.2f} kg/m2 fresh")
    if result.solve_success is not None:
        print(f"  solve success:      {result.solve_success:.4f}")
    print(f"  decision time:      {result.online_cost_seconds * 1000:.1f} ms/step")
    print("  outside the allowed band (hours; size x hours; worst step):")
    for name, exceedance in result.exceedances.items():
        profile = safety_profile_from_exceedance([exceedance], step_hours=step_hours)
        unit = _UNITS.get(name, "")
        print(
            f"    {name:>11}: {profile.mean_duration_h:6.1f} h  "
            f"{profile.degree_hours:7.1f} {unit}.h  worst {profile.worst_excursion:.2f} {unit}"
        )
