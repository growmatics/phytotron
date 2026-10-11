"""A minimal custom controller: a thermostat with ventilation, run for one simulated season.

This is the template for ``docs/own-controller.md``. Copy it, change ``act``, and run:

    GLG2_WEATHER_DIR=weather python examples/my_controller.py --days 7

It heats when the air is below a setpoint and ventilates when it is above another, with
everything else off. It is deliberately naive and not a good controller.
"""

from __future__ import annotations

if (
    __name__ == "__main__"
):  # as a script: pin before numpy loads; as an imported module: the runner did
    from phytotron.reproducibility import pin_threads

    pin_threads()

import argparse
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from phytotron.config import load_config
from phytotron.controllers.base import ControlContext
from phytotron.envs.glg2 import (
    make_benchmark_env,
    paths_from_env,
    run_season_controller,
)

# GreenLight-Gym2's control vector, each entry in [0, 1].
HEATING, CO2, THERMAL_SCREEN, VENTILATION, LAMPS, BLACKOUT_SCREEN = range(6)
AIR_TEMP = 2  # index of air temperature [C] in the 28-entry state


class Thermostat:
    """Proportional heating below ``heat_c`` and ventilation above ``vent_c``."""

    control_kind = "absolute"  # returns controls, not increments (see the docs)

    def __init__(self, heat_c: float = 18.0, vent_c: float = 24.0, band_c: float = 2.0) -> None:
        self.heat_c, self.vent_c, self.band_c = heat_c, vent_c, band_c

    def reset(self, seed: int) -> None:
        """No internal state; a stateful controller clears it here."""

    def act(self, ctx: ControlContext) -> NDArray[np.float64]:
        t_air = float(ctx.state[AIR_TEMP])
        u = np.zeros(6)
        u[HEATING] = np.clip((self.heat_c - t_air) / self.band_c, 0.0, 1.0)
        u[VENTILATION] = np.clip((t_air - self.vent_c) / self.band_c, 0.0, 1.0)
        return u


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--days", type=int, default=None, help="default: config season length")
    args = parser.parse_args()

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml")
    gen = cfg.generalisation
    days = args.days or gen.season_length_days
    config_dir, weather_dir = paths_from_env(os.environ)
    env = make_benchmark_env(
        cfg, args.year, config_dir=config_dir, weather_dir=weather_dir, season_days=days
    )
    result = run_season_controller(env, Thermostat(), seed=0)
    print(f"thermostat, simulated season, {gen.location} {args.year}, {days} days")
    print(f"  net economic value: {result.net_economic_value:+.3f} EUR/m2")
    step_hours = float(cfg.raw["episode"]["timestep_seconds"]) / 3600.0
    for name, exc in result.exceedances.items():
        print(f"  {name:>11}: {float((exc > 0).sum()) * step_hours:6.1f} h outside the band")


if __name__ == "__main__":
    main()
