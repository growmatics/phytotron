"""L5: reproduce van Laatum et al.'s standalone-MPC numbers (economic MPC validation).

Runs the physics-only economic MPC (``objective="economic"``) on the paper's exact
config (Bleiswijk 2023, 1 Jun; dt=300 s; N=12; T=288) and scores it with GreenLight-Gym2's
own economic reward, then compares to the reported standalone-MPC EPI / penalty / objective
(``l5_reproduction.targets`` in ``configs/benchmark.yaml``). See ``docs/validation.md``
and ``docs/mpc.md``.

    GLG2_WEATHER_DIR=<GreenLight-Gym2 v0.2.1 checkout>/gl_gym/environments/weather \
    python examples/l5_reproduce.py

The paper's weather ships with GreenLight-Gym2 as ``Bleiswijk/KASPRO2023.csv``. The
loader expects ``Bleiswijk/2023.csv``, so this script copies the file into a temporary
folder and points the simulator there. The checkout is only read.
"""

from __future__ import annotations

from phytotron.reproducibility import pin_threads

pin_threads()  # before numpy loads; MPC results depend on it

import os  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from gl_gym.environments.tomato_env import TomatoEnv  # noqa: E402

from phytotron.config import load_config  # noqa: E402
from phytotron.controllers.base import NominalParameters  # noqa: E402
from phytotron.controllers.mpc import PhysicsOnlyMPC  # noqa: E402
from phytotron.envs.glg2 import paths_from_env  # noqa: E402

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "benchmark.yaml"
_ACC_FIELDS = (
    "EPI",
    "revenue",
    "heat_cost",
    "co2_cost",
    "elec_cost",
    "temp_violation",
    "co2_violation",
    "rh_violation",
)


def _provision_weather(
    source_dir: str, target_dir: str, location: str, year: int, source_file: str
) -> None:
    """Copy the shipped ``<source_file>`` to ``<target_dir>/<location>/<year>.csv``."""
    dest = Path(target_dir) / location / f"{year}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source_dir) / location / source_file, dest)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="phytotron-l5-") as weather_dir:
        _run(str(paths_from_env(os.environ)[1]), weather_dir)


def _run(source_weather_dir: str, weather_dir: str) -> None:
    cfg = load_config(str(_CONFIG))
    ec = cfg.l5_economic()
    l5 = cfg.raw["l5_reproduction"]
    bounds = {k: tuple(v) for k, v in l5["bounds"].items()}
    dt = float(l5["timestep_seconds"])
    horizon = int(l5["horizon_steps"])
    sim_steps = int(l5["sim_steps"])
    w = l5["weather"]
    _provision_weather(source_weather_dir, weather_dir, w["location"], int(w["year"]), w["file"])

    base = {
        "weather_data_dir": weather_dir,
        "location": w["location"],
        "num_params": 208,
        "nx": 28,
        "nu": 6,
        "nd": 10,
        "dt": dt,
        "u_min": [0] * 6,
        "u_max": [1] * 6,
        "delta_u_max": 0.1,
        "pred_horizon": 0.5,
        "season_length": 1,
        "start_train_year": int(w["year"]),
        "end_train_year": int(w["year"]),
        "start_train_day": int(w["day_of_year"]),
        "end_train_day": int(w["day_of_year"]),
        "training": True,
    }
    env = TomatoEnv(
        reward_function="GreenhouseReward",
        observation_modules=[
            "IndoorClimateObservations",
            "BasicCropObservations",
            "ControlObservations",
            "WeatherObservations",
            "TimeObservations",
            "WeatherForecastObservations",
        ],
        constraints={
            "co2_min": bounds["co2_ppm"][0],
            "co2_max": bounds["co2_ppm"][1],
            "temp_min": bounds["air_temp_c"][0],
            "temp_max": bounds["air_temp_c"][1],
            "rh_min": bounds["rh_percent"][0],
            "rh_max": bounds["rh_percent"][1],
        },
        eval_options={
            "eval_days": [int(w["day_of_year"])],
            "eval_years": [int(w["year"])],
            "location": w["location"],
        },
        reward_params={
            "fixed_greenhouse_cost": 0.0,
            "fixed_co2_cost": 0.0,
            "fixed_lamp_cost": 0.0,
            "fixed_screen_cost": 0.0,
            "elec_price": ec.elec_price,
            "heating_price": ec.heating_price,
            "co2_price": ec.co2_price,
            "fruit_price": ec.fruit_price,
            "dmfm": ec.dmfm,
            "pen_weights": list(ec.penalty_weights),
            "pen_lamp": 0.1,
        },
        base_env_params=base,
    )
    env.reset(seed=0)
    params = NominalParameters({str(i): float(v) for i, v in enumerate(np.array(env.p).flatten())})
    mpc = PhysicsOnlyMPC(
        params, horizon=horizon, dt=dt, objective="economic", economic=ec, climate_bounds=bounds
    )
    mpc.reset()

    acc = dict.fromkeys(_ACC_FIELDS, 0.0)
    lam = ec.penalty_weights
    penalty = 0.0
    solve_times = []
    t0 = time.time()
    for _ in range(sim_steps):
        x = np.array(env.x).flatten()
        forecast = np.array(env.weather_data[env.timestep : env.timestep + horizon + 1])
        a = time.time()
        u = mpc.solve(x, forecast)
        solve_times.append(time.time() - a)
        # absolute controls: apply directly (env.step would treat u as an increment)
        _, _, _, _, info = env.step_raw_control(u)
        for k in _ACC_FIELDS:
            acc[k] += info[k]
        penalty += (
            lam[0] * info["co2_violation"]
            + lam[1] * info["temp_violation"]
            + lam[2] * info["rh_violation"]
        )
    wall = time.time() - t0
    st = np.array(solve_times)
    tgt = l5["targets"]
    epi, obj = acc["EPI"], acc["EPI"] - penalty
    print(f"=== L5 standalone-MPC reproduction ({sim_steps} steps, {wall:.0f}s wall) ===")
    print(f"solve: median {np.median(st):.2f}s  mean {st.mean():.2f}s  max {st.max():.2f}s")
    print(f"EPI          = {epi:+.4f}   (paper {tgt['epi']:+.3f})")
    print(f"penalty      = {penalty:.4f}   (paper {tgt['penalty']:.3f})")
    print(f"objective J  = {obj:+.4f}   (paper {tgt['objective']:+.3f})")
    print(f"revenue      = {acc['revenue']:.4f}   (paper ~0.05; low = the myopia signature)")
    print(
        f"costs: heat {acc['heat_cost']:.4f}  "
        f"co2 {acc['co2_cost']:.4f}  elec {acc['elec_cost']:.4f}"
    )
    print(
        "penalty weighted: "
        f"temp {lam[1] * acc['temp_violation']:.4f}  "
        f"co2 {lam[0] * acc['co2_violation']:.4f}  "
        f"rh {lam[2] * acc['rh_violation']:.4f}"
    )


if __name__ == "__main__":
    main()
