"""Build the reference controllers from the config: one place, used by every script.

``build_controller(name, env, cfg, ...)`` returns a ready :class:`Controller` for one
season. The runner, the quickstart and the reproduction scripts all call it, so the MPC
setup (``mpc`` block of ``configs/benchmark.yaml``) and the incumbent's parameters
(GreenLight-Gym2's ``rule_based.yml``) are read in one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from phytotron.config import BenchmarkConfig
from phytotron.controllers.base import NominalParameters
from phytotron.controllers.mpc import PhysicsOnlyMPC
from phytotron.controllers.rule_based import RuleBasedIncumbent
from phytotron.mismatch import perturb_params

ARMS = ("incumbent", "physics_mpc")
MODEL_BASED_ARMS = frozenset({"physics_mpc"})
DETERMINISTIC_ARMS = frozenset({"incumbent"})


def incumbent_params(glg2_config_dir: str | Path) -> dict[str, Any]:
    """The incumbent's tuned parameters, from GreenLight-Gym2's ``rule_based.yml``."""
    path = Path(glg2_config_dir).parent / "agents" / "rule_based.yml"
    return dict(yaml.safe_load(path.read_text())["TomatoEnv"])


def mpc_settings(cfg: BenchmarkConfig) -> dict[str, Any]:
    """The MPC arm's constructor settings, except the nominal model."""
    if cfg.mpc is None or cfg.economic is None:
        raise ValueError("configs/benchmark.yaml needs the mpc and economic blocks")
    band = cfg.raw["constraints"]
    return {
        "horizon": cfg.mpc.horizon_steps,
        "dt": float(cfg.raw["episode"]["timestep_seconds"]),
        "objective": "economic",
        "economic": cfg.economic,
        "climate_bounds": {k: tuple(band[k]) for k in ("co2_ppm", "air_temp_c", "rh_percent")},
        "terminal_weight": cfg.mpc.terminal_weight,
    }


def build_controller(
    name: str,
    env: Any,
    cfg: BenchmarkConfig,
    *,
    glg2_config_dir: str | Path,
    mismatch: float = 0.0,
    seed: int = 0,
) -> Any:  # pragma: no cover - needs gl_gym
    """A reference controller for one season on ``env``.

    A model-based arm gets a nominal model: the plant's parameters perturbed by
    ``mismatch`` with ``seed``, never the plant's own (the inverse-crime boundary).
    """
    if name == "incumbent":
        return RuleBasedIncumbent(incumbent_params(glg2_config_dir))
    if name == "physics_mpc":
        true = {str(i): float(v) for i, v in enumerate(np.array(env.p).flatten())}
        nominal = NominalParameters(perturb_params(true, mismatch, seed))
        return PhysicsOnlyMPC(nominal, **mpc_settings(cfg))
    raise ValueError(f"unknown arm {name!r}; available: {list(ARMS)}")
