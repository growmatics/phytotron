"""Synthetic season records and a fake environment for the tests, no simulator."""

from pathlib import Path

import numpy as np

from phytotron.config import load_config
from phytotron.envs.glg2 import SeasonResult
from phytotron.grid import Task
from phytotron.records import FailureRecord, SeasonRecord, season_record

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "benchmark.yaml"
CFG = load_config(CONFIG)


def _record(
    arm: str,
    level: float,
    seed: int,
    net: float,
    rh: list[float],
    year: int = 2025,
    climate: dict[str, list[float]] | None = None,
) -> SeasonRecord:
    task = Task(arm, year, level, seed, 60)
    result = SeasonResult(
        net_economic_value=net,
        exceedances={
            "humidity": np.array(rh),
            "temperature": np.zeros(len(rh)),
            "co2": np.zeros(len(rh)),
        },
        online_cost_seconds=0.1 if arm == "physics_mpc" else 0.0,
        solve_success=1.0 if arm == "physics_mpc" else None,
        revenue=net + 1.6,
        heating_cost=1.0,
        co2_cost=0.4,
        electricity_cost=0.2,
        steps=len(rh),
        climate=None if climate is None else {k: np.array(v) for k, v in climate.items()},
    )
    return season_record(task, result)


def _calm(steps: int = 96) -> dict[str, list[float]]:
    return {"temperature": [18.0] * steps, "humidity": [80.0] * steps, "radiation": [0.0] * steps}


class _Env:
    def reset(self, seed: int = 0) -> None:
        self.seed = seed


def _fake_season(steps: int) -> SeasonResult:
    return SeasonResult(
        net_economic_value=1.0,
        exceedances={"humidity": np.zeros(steps), "temperature": np.zeros(steps)},
        online_cost_seconds=0.0,
        solve_success=None,
        steps=steps,
        max_action_excess=0.0,
    )


def _failed(level: float, seed: int, stage: str | None = None) -> FailureRecord:
    """A failure record as an older run wrote it: no traceback, and no stage unless given."""
    task = {"arm": "physics_mpc", "year": 2025, "mismatch": level, "seed": seed, "season_days": 60}
    record = {"task": task, "label": f"2025/m{level:.2f}", "error": "RuntimeError: x"}
    return FailureRecord.from_json(record if stage is None else {**record, "stage": stage})
