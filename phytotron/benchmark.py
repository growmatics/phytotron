"""Aggregation of a benchmark run: the gates, the safety profiles and the Pareto front.

:class:`BenchmarkResult` holds every completed season of a run, keyed by arm, condition
label and seed, with the counts of seasons that failed or never started, and derives the
per-arm numbers on demand: net economic value, the season-level feasibility gate under
each of its three rules, the per-class safety profile, decision time and the Pareto
front. :mod:`phytotron.summary` builds it from a run directory's season records;
``run_season_controller`` in ``envs/glg2.py`` is the primitive that produces one season.
Every number is derived from the stored seasons, so the raw records stay the source of
truth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from phytotron.config import GateConfig
from phytotron.envs.glg2 import SeasonResult
from phytotron.metrics.crop_rules import broken_rules
from phytotron.metrics.gate import clopper_pearson_upper, passes_soft_gate
from phytotron.metrics.pareto import hypervolume_2d, normalize_minmax, pareto_front
from phytotron.metrics.safety import (
    SafetyProfile,
    banded_violation,
    safety_profile_from_exceedance,
)

_MIN_PARETO_ARMS = 2  # need at least the incumbent plus one comparator for a front


@dataclass(frozen=True)
class GateCount:
    """One class's gate result for one arm: failing seasons (crashes included), the
    seasons counted, the one-sided Clopper-Pearson upper bound on the failing rate, and
    whether that bound is within epsilon."""

    failing: int
    seasons: int
    upper_bound: float
    passes: bool


@dataclass(frozen=True)
class BenchmarkResult:
    """Every completed season of a run, plus the gate settings and the failure counts.

    Keyed by ``(arm, season label, seed)``. Aggregations (net value, the three gates,
    safety, decision time, the Pareto front) are derived on demand so the raw per-season
    results stay the source of truth.
    """

    seasons: dict[tuple[str, str, int], SeasonResult]
    gate: GateConfig
    soft_classes: tuple[str, ...]  # exceedance classes gated as soft crop-health
    step_hours: float
    # Seasons that CRASHED (per arm) while the controller was built, during the season or
    # in its checks. Such a season is a deployment failure, not a missing observation: it
    # counts as violating every soft class in the gate (never silently excluded, which
    # would bias toward the arm that crashes on its hardest seasons). Net-value stats are
    # over completed seasons only, reported next to these counts.
    failed: dict[str, int] = field(default_factory=dict)
    # Seasons that never reached the controller (per arm): the harness failed in its own
    # setup (config, environment, reset). Listed, never counted: they are in no gate
    # count, so the arm's season count drops and the underpowered check sees it.
    not_started: dict[str, int] = field(default_factory=dict)

    def arms(self) -> list[str]:
        known = {arm for (arm, _c, _s) in self.seasons}
        return sorted(known | set(self.failed) | set(self.not_started))

    def _arm_seasons(self, arm: str) -> list[SeasonResult]:
        return [r for (a, _c, _s), r in self.seasons.items() if a == arm]

    def net_value_summary(self, arm: str) -> tuple[float | None, float | None]:
        """Mean and standard deviation of net economic value over the arm's COMPLETED
        seasons (a crashed season has no honest number; report it via ``failed`` and the
        gate instead of inventing one). ``(None, None)`` without a completed season: a
        summary never carries NaN, which is not JSON."""
        vals = np.array([r.net_economic_value for r in self._arm_seasons(arm)])
        if vals.size == 0:
            return None, None
        return float(vals.mean()), float(vals.std())

    def _gate(
        self, arm: str, violates: Callable[[SeasonResult, str], bool]
    ) -> dict[str, GateCount]:
        """The season-level gate per soft class, under one per-season rule.

        ``violates(season, cls)`` says whether a completed season violates the class. A
        season that CRASHED counts as violating every class (conservative: an un-runnable
        controller cannot claim feasibility on the season it failed); a season the
        harness could not start is in ``not_started`` and in no count. The one-sided
        Clopper-Pearson upper bound on the violating-season rate must be at most epsilon.
        """
        seasons = self._arm_seasons(arm)
        failed = self.failed.get(arm, 0)
        n = len(seasons) + failed
        counts = {}
        for cls in self.soft_classes:
            k = failed + sum(1 for r in seasons if violates(r, cls))
            counts[cls] = GateCount(
                failing=k,
                seasons=n,
                upper_bound=clopper_pearson_upper(k, n, self.gate.delta) if n else 1.0,
                passes=passes_soft_gate(k, n, self.gate.epsilon, self.gate.delta),
            )
        return counts

    def strict_gate(self, arm: str) -> dict[str, GateCount]:
        """A season violates a class if that class is exceeded at any step."""
        return self._gate(arm, lambda r, cls: bool(r.exceedances.get(cls, np.zeros(1)).max() > 0.0))

    def banded_gate(self, arm: str, bands: dict[str, tuple[float, float]]) -> dict[str, GateCount]:
        """A season violates a class only if it stays more than ``depth`` outside its bound
        for more than ``min_hours`` consecutive hours, ``bands[cls] = (depth, min_hours)``."""
        return self._gate(
            arm,
            lambda r, cls: (
                cls in r.exceedances
                and banded_violation(r.exceedances[cls], self.step_hours, *bands[cls])
            ),
        )

    def crop_gate(
        self, arm: str, step_seconds: int, rules: dict[str, dict[str, dict[str, Any]]]
    ) -> dict[str, GateCount] | None:
        """A season violates a class if its climate trace breaks any of ``rules[cls]`` (see
        :mod:`phytotron.metrics.crop_rules`). None if any completed season has no climate
        trace: the gate cannot be computed, and the Pareto ranking treats the arm as not
        a passer."""
        if any(r.climate is None for r in self._arm_seasons(arm)):
            return None
        return self._gate(arm, lambda r, cls: _breaks_crop_rules(r, cls, step_seconds, rules[cls]))

    def gate_underpowered(self, arm: str) -> bool:
        """True when the arm's season count (completed + failed) is below the
        configured ``n_seasons_min`` floor, so its gate outcome is off-protocol
        and must be reported as such (never silently)."""
        n = len(self._arm_seasons(arm)) + self.failed.get(arm, 0)
        return n < self.gate.n_seasons_min

    def safety(self, arm: str, cls: str) -> SafetyProfile:
        """Pooled per-class safety profile across the arm's seasons."""
        traces = [r.exceedances[cls] for r in self._arm_seasons(arm) if cls in r.exceedances]
        return safety_profile_from_exceedance(traces, step_hours=self.step_hours)

    def decision_time(self, arm: str) -> float | None:
        """Mean online (deployment) decision time per step [s] over the arm's completed
        seasons; None without one (never NaN).

        Wall-clock, hence hardware-dependent: valid as a *relative* Pareto axis on one
        machine, but interpret the absolute value only with the run record's hardware
        stamp (see :func:`hardware_stamp`).
        """
        seasons = self._arm_seasons(arm)
        return float(np.mean([r.online_cost_seconds for r in seasons])) if seasons else None

    def solve_success(self, arm: str) -> float | None:
        """Mean MPC solve-success over the arm's seasons (deployment reliability).

        Fraction of NLP solves that converged or salvaged a usable iterate. ``None`` for
        solver-free arms (rule-based, RL), whose seasons carry no ``solve_success``.
        """
        vals = [r.solve_success for r in self._arm_seasons(arm) if r.solve_success is not None]
        return float(np.mean(vals)) if vals else None

    def pareto(
        self, incumbent: str = "incumbent", gate_passers_only: bool = True
    ) -> dict[str, Any]:
        """The (net value, online cost) Pareto comparison over gate-passing arms.

        Maximises net economic value and minimises online compute; min-max normalises each
        axis. The hypervolume is measured against the anti-ideal corner (worst on each axis),
        the only valid reference: the incumbent is typically cost-optimal and so cannot anchor
        a hypervolume (see ``metrics/pareto.py``). ``hypervolume_vs_incumbent`` reports the
        improvement over the deployed status quo, HV(front) - HV(incumbent), both against the
        anti-ideal reference. The hypervolume mixes value and cost on one
        normalised scale, so treat it as a diagnostic, not a ranking input. The incumbent is
        always included even if it fails the gate. Returns the considered arms, the
        non-dominated set, and the two hypervolumes. An arm without a completed season has
        no point to rank and is left out.
        """
        arms = [
            a
            for a in self.arms()
            if self._arm_seasons(a)
            and (not gate_passers_only or all(g.passes for g in self.strict_gate(a).values()))
        ]
        if incumbent in self.arms() and incumbent not in arms:
            arms = [*arms, incumbent]
        arms = sorted(arms)
        if len(arms) < _MIN_PARETO_ARMS or incumbent not in arms:
            return {
                "arms": arms,
                "non_dominated": arms,
                "hypervolume": 0.0,
                "hypervolume_vs_incumbent": 0.0,
                "reference": "anti_ideal",
            }
        # maximisation on both axes: net value up, online cost down (negated)
        costs = {a: self.decision_time(a) for a in arms}
        assert all(c is not None for c in costs.values())  # arms have completed seasons
        points = np.array([[self.net_value_summary(a)[0], -(costs[a] or 0.0)] for a in arms])
        norm, _lo, _hi = normalize_minmax(points)
        non_dominated = [arms[i] for i in pareto_front(norm)]
        # A valid hypervolume reference is the anti-ideal corner (worst on each axis); in
        # min-max-normalised coordinates that is the origin. The incumbent cannot anchor it:
        # it is typically cost-optimal, so nothing dominates it on cost and HV collapses to 0.
        reference = np.zeros(points.shape[1])
        inc = arms.index(incumbent)
        hv_front = float(hypervolume_2d(norm, reference))
        hv_incumbent = float(hypervolume_2d(norm[inc : inc + 1], reference))
        return {
            "arms": arms,
            "non_dominated": non_dominated,
            "hypervolume": hv_front,
            "hypervolume_vs_incumbent": hv_front - hv_incumbent,
            "reference": "anti_ideal",
        }


def _breaks_crop_rules(
    season: SeasonResult, cls: str, step_seconds: int, rules: dict[str, dict[str, Any]]
) -> bool:
    """True if the season's climate trace breaks any of the class's crop rules."""
    if season.climate is None:
        return False
    return bool(
        broken_rules(season.climate[cls], step_seconds, rules, season.climate.get("radiation"))
    )
