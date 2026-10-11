"""Check that Phytotron reproduces one season of the reference run's economic MPC (physics_mpc).

"The study" is Growmatics' reference run of 24 July 2026 (``docs/validation.md``, section 1).
Runs ``PhysicsOnlyMPC`` with the study's arm setup (the ``mpc`` block of the config:
horizon 4, terminal weight 0; economic objective; nominal model from ``perturb_params``)
on held-out year 2025 at mismatch 0.00, seed 0, and compares the net economic value with
the study's recorded benchmark run (Growmatics, forthcoming).

The math libraries are pinned to one thread, as in the study's run. With the default
thread count the same season gave -1.745 instead of -1.608 EUR/m2: multithreaded sums
round differently, and over 5760 solves that moves the solver onto a different path.
With one thread the result matches exactly. One season takes about 20 to 25 minutes.

    GLG2_WEATHER_DIR=<dir containing Amsterdam_OM/> python examples/reproduce_study_mpc.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from phytotron.reproducibility import pin_threads

pin_threads()  # before numpy loads; MPC results depend on it (see docstring)

from phytotron.arms import build_controller  # noqa: E402
from phytotron.config import load_config  # noqa: E402
from phytotron.envs.glg2 import (  # noqa: E402
    make_benchmark_env,
    paths_from_env,
    run_season_controller,
)

# Net economic value [EUR/m2] of physics_mpc, the study's recorded run (2025/m0.00, seed 0).
YEAR, MISMATCH, SEED = 2025, 0.0, 0
STUDY_MPC_NET = -1.6076154242219172
TOLERANCE = 1e-6


def main() -> int:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml")
    config_dir, weather_dir = paths_from_env(os.environ)
    env = make_benchmark_env(cfg, YEAR, config_dir=config_dir, weather_dir=weather_dir)
    env.reset(seed=SEED)
    mpc = build_controller(
        "physics_mpc", env, cfg, glg2_config_dir=config_dir, mismatch=MISMATCH, seed=SEED
    )
    result = run_season_controller(env, mpc, SEED)
    diff = result.net_economic_value - STUDY_MPC_NET
    ok = abs(diff) <= TOLERANCE
    print(
        f"{YEAR}/m{MISMATCH:.2f} seed {SEED}: phytotron {result.net_economic_value:+.6f}  "
        f"study {STUDY_MPC_NET:+.6f}  diff {diff:+.2e}  "
        f"solve success {result.solve_success}  {'OK' if ok else 'MISMATCH'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
