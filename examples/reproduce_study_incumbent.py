"""Check that Phytotron reproduces the reference run's rule-based incumbent, season by season.

"The study" is Growmatics' reference run of 24 July 2026 (``docs/validation.md``, section 1).
Runs the incumbent on the four held-out weather years with the study's season setup
(location, start day and season length from ``configs/benchmark.yaml``, seed 0) and compares
the net economic value with the numbers recorded in the study's benchmark run
(Growmatics, forthcoming). The incumbent is
deterministic, so the values must match to numerical precision.

    GLG2_WEATHER_DIR=<dir containing Amsterdam_OM/> python examples/reproduce_study_incumbent.py

Phytotron ships the simulator's configuration files (``phytotron/scenarios``); set
``GLG2_CONFIG_DIR`` only to run against a GreenLight-Gym2 checkout instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from phytotron.arms import build_controller
from phytotron.config import load_config
from phytotron.envs.glg2 import make_benchmark_env, paths_from_env, run_season_controller

# Net economic value [EUR/m2] of the incumbent per held-out year, the study's recorded run.
STUDY_INCUMBENT_NET = {
    2022: 2.044325875964997,
    2023: -9.444210990371449,
    2024: -6.654230083654138,
    2025: 3.0875140901195106,
}
TOLERANCE = 1e-6


def main() -> int:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml")
    config_dir, weather_dir = paths_from_env(os.environ)
    failures = 0
    for year, expected in STUDY_INCUMBENT_NET.items():
        env = make_benchmark_env(cfg, year, config_dir=config_dir, weather_dir=weather_dir)
        env.reset(seed=0)
        incumbent = build_controller("incumbent", env, cfg, glg2_config_dir=config_dir)
        result = run_season_controller(env, incumbent, seed=0)
        diff = result.net_economic_value - expected
        ok = abs(diff) <= TOLERANCE
        failures += not ok
        print(
            f"{year}: phytotron {result.net_economic_value:+.6f}  study {expected:+.6f}  "
            f"diff {diff:+.2e}  {'OK' if ok else 'MISMATCH'}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
