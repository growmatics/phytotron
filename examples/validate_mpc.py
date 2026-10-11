"""Economic MPC validation, layers L2 and L3 (see ``docs/validation.md``).

Layer L1 (the collocation coefficients) lives in ``tests/test_mpc.py`` and runs in CI.
This script runs two integration layers against the real simulator, on the controller
the benchmark runs: built through the same factory as a benchmark season, so the
objective, the horizon and the discretisation are the benchmark's.

  L2  prediction-model fidelity: drive the MPC's own one-interval model (the one its
      OCP plans with) and the CVODES plant with the SAME controls and quantify divergence.
  L3  optimizer status: does the MPC solve, return valid controls, respect bounds.

L5 (reproduce the published numbers) is ``examples/l5_reproduce.py``. Requires the
fetched weather.

    GLG2_WEATHER_DIR=weather python examples/validate_mpc.py
"""

from __future__ import annotations

from phytotron.reproducibility import pin_threads

pin_threads()  # before numpy loads

import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from phytotron.arms import build_controller  # noqa: E402
from phytotron.config import load_config  # noqa: E402
from phytotron.envs.glg2 import make_tomato_env, paths_from_env  # noqa: E402

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"
VALIDATION_YEAR = 2018  # a training year; L2 and L3 are not scored
ROLLOUT_STEPS = 6  # L2: how many intervals the model and the plant are driven
_LABELS = {0: "co2Air", 2: "tAir", 15: "vpAir", 25: "cFruit"}
_BOUND_TOL = 1e-6


def main() -> None:
    cfg = load_config(_CONFIG)
    gen = cfg.generalisation
    config_dir, weather_dir = paths_from_env(os.environ)
    env = make_tomato_env(
        config_dir,
        weather_dir,
        overrides={
            "location": gen.location,
            "start_train_year": VALIDATION_YEAR,
            "end_train_year": VALIDATION_YEAR,
            "start_train_day": gen.season_start_day,
            "end_train_day": gen.season_start_day,
            "training": True,
        },
    )
    env.reset(seed=0)
    # The benchmark's own MPC: mismatch 0 gives it the plant's parameters as its nominal
    # model, so L2 measures the discretisation alone, not model error.
    mpc = build_controller("physics_mpc", env, cfg, glg2_config_dir=config_dir, mismatch=0.0)
    print(
        f"Controller under test: objective {mpc.objective}, horizon {mpc.horizon} steps of "
        f"{mpc.dt:.0f} s, one Radau element of degree {mpc.degree} per step"
    )

    control = np.array([0.3, 0.2, 1.0, 0.0, 0.0, 0.0])  # mild fixed control
    x0 = np.array(env.x).flatten()
    weather = [np.array(env.weather_data[env.timestep + k]) for k in range(ROLLOUT_STEPS)]

    # L2: roll the plant and the MPC model from the same x0 under the same controls
    plant = [x0.copy()]
    xp = x0.copy()
    for _ in range(ROLLOUT_STEPS):
        _, _, _, _, _ = env.step_raw_control(control)
        xp = np.array(env.x).flatten()
        plant.append(xp.copy())
    model = [x0.copy()]
    xm = x0.copy()
    for k in range(ROLLOUT_STEPS):
        xm = mpc.model_step(xm, control, weather[k])
        model.append(xm.copy())

    print("=== L2 prediction-model fidelity (model vs CVODES plant, same controls) ===")
    for idx, name in _LABELS.items():
        pv = np.array([p[idx] for p in plant])
        mv = np.array([m[idx] for m in model])
        rel = np.abs(mv - pv) / (np.abs(pv) + 1e-9)
        print(f"  {name:>7}: max abs err {np.abs(mv - pv).max():.4g}  max rel err {rel.max():.2%}")

    # L3: optimizer status on a few steps
    print("\n=== L3 optimizer status (does the MPC solve, bounds respected) ===")
    env.reset(seed=0)
    mpc.reset()
    solved = 0
    for t in range(4):
        fc = np.array(env.weather_data[env.timestep : env.timestep + mpc.horizon + 1])
        try:
            u = mpc.solve(np.array(env.x).flatten(), fc)
            ok = bool(np.all(u >= -_BOUND_TOL) and np.all(u <= 1 + _BOUND_TOL))
            solved += ok
            env.step_raw_control(u)
            print(f"  step {t}: solved, controls in bounds={ok}")
        except Exception as exc:  # report, do not crash the harness
            print(f"  step {t}: SOLVE FAILED ({type(exc).__name__})")
            break
    print(f"L3: {solved}/4 steps solved with valid controls")


if __name__ == "__main__":
    main()
