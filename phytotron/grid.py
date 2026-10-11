"""The benchmark grid: which seasons run, as plain-data tasks.

Axes per arm, from ``configs/benchmark.yaml``: every arm runs every held-out weather year;
only model-based arms (``physics_mpc``) sweep the mismatch ladder, since the others have no
model to perturb; a season runs one seed only when seeds cannot change it (a deterministic
arm on this deterministic simulator, or a model-based arm at mismatch 0, whose nominal
model is then the plant's own for every seed). More seeds would add identical seasons, not
independent trials. :func:`pending` lists what a resumed run still has to do.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from phytotron.arms import (
    ARMS,
    DETERMINISTIC_ARMS,
    MODEL_BASED_ARMS,
)
from phytotron.config import BenchmarkConfig
from phytotron.controllers.registry import ArmSpec


@dataclass(frozen=True)
class Task:
    """One season to run: an arm on one weather year, mismatch level and seed."""

    arm: str
    year: int
    mismatch: float
    seed: int
    season_days: int
    controller: str | None = None  # "module:attr" for a user controller (see registry)

    @property
    def label(self) -> str:
        """Condition label, e.g. ``2025/m0.10`` (the format of the recorded runs)."""
        return f"{self.year}/m{self.mismatch:.2f}"

    @property
    def key(self) -> str:
        """File-safe identifier, unique within one run. Keep this format stable: run
        directories written by earlier versions are read back by it."""
        return f"{self.arm}__{self.year}__m{self.mismatch:.3f}__s{self.seed}"


def seed_invariant(
    arm: str, mismatch: float, registered: Mapping[str, ArmSpec] | None = None
) -> bool:
    """True when the seed cannot change the season, so one seed is enough."""
    if registered and arm in registered:
        return registered[arm].seed_invariant(mismatch)
    return arm in DETERMINISTIC_ARMS or (arm in MODEL_BASED_ARMS and mismatch == 0.0)


def is_model_based(arm: str, registered: Mapping[str, ArmSpec] | None = None) -> bool:
    """Model-based arms sweep the mismatch ladder; their verdict pools mismatch above 0."""
    if registered and arm in registered:
        return registered[arm].model_based
    return arm in MODEL_BASED_ARMS


def build_tasks(
    cfg: BenchmarkConfig,
    arms: Sequence[str] = ARMS,
    *,
    years: Sequence[int] | None = None,
    mismatch_levels: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
    season_days: int | None = None,
    registered: Mapping[str, ArmSpec] | None = None,
) -> list[Task]:
    """The grid for ``arms``, with config defaults for every axis not given.

    ``registered`` maps the names of user controllers (``--controller``) to their specs; a
    registered arm's tasks carry its ``module:attr`` target so workers can build it.
    """
    registered = dict(registered or {})
    known = [*ARMS, *registered]
    unknown = sorted(set(arms) - set(known))
    if unknown:
        raise ValueError(f"unknown arms {unknown}; available: {known}")
    years = list(years or cfg.generalisation.heldout_years)
    levels = list(mismatch_levels or cfg.grid.mismatch_levels)
    seeds = list(seeds or range(cfg.grid.seeds_per_condition))
    days = season_days or cfg.generalisation.season_length_days
    tasks = []
    for arm in arms:
        arm_levels = levels if is_model_based(arm, registered) else [0.0]
        target = registered[arm].target if arm in registered else None
        for year in years:
            for level in arm_levels:
                level_seeds = seeds[:1] if seed_invariant(arm, float(level), registered) else seeds
                for seed in level_seeds:
                    tasks.append(Task(arm, int(year), float(level), int(seed), int(days), target))
    return tasks
