"""Aggregation of a benchmark run: the gates, the safety profiles and the Pareto front.

:class:`BenchmarkResult` holds every completed season of a run, keyed by arm, condition
label and seed, with the counts of seasons that failed or never started, and derives the
per-arm numbers on demand: net economic value, the season-level feasibility gate under
each of its three rules, the per-class safety profile, decision time and the Pareto
front. :mod:`phytotron.runner` builds it from a run directory's season records;
``run_season_controller`` in ``envs/glg2.py`` is the primitive that produces one season.
Every number is derived from the stored seasons, so the raw records stay the source of
truth.
"""

from __future__ import annotations

import os
import platform
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
from phytotron.reproducibility import thread_settings

_MIN_PARETO_ARMS = 2  # need at least the incumbent plus one comparator for a front


def hardware_stamp() -> dict[str, Any]:
    """Machine identity for the run record, so the online-cost axis is interpretable.

    ``online_cost`` is wall-clock time (``time.perf_counter``), so it is **relative**:
    comparable across arms on the *same* machine (the Pareto ordering is valid), but the
    absolute ms/step is not portable across hardware. Stamping the machine makes the
    numbers reproducible in context. A fully hardware-independent cost proxy (IPOPT
    iteration count / NLP evaluations per step) is a possible future refinement.

    The math-library thread settings are recorded too: MPC results depend on them
    (see :mod:`phytotron.reproducibility`).
    """
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "threads": thread_settings(),
    }


@dataclass(frozen=True)
class BenchmarkResult:
    """Every season's result across the grid, plus the run record and gate settings.

    Keyed by ``(arm, condition_label, seed)``. Aggregations (net value, gate, safety)
    are derived on demand so the raw per-season results stay the source of truth.
    """

    seasons: dict[tuple[str, str, int], SeasonResult]
    record: dict[str, Any]  # config version, git commit, grid, timestamp -- traceability
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

    def gate_outcome(self, arm: str) -> dict[str, bool]:
        """Per soft class: does the arm clear the season-level feasibility gate?

        A season counts as violating a class if that class is exceeded at any step; the
        Clopper-Pearson upper bound on the violating-season rate must be <= epsilon.
        A season that CRASHED counts as violating every class (conservative: an
        un-runnable controller cannot claim feasibility on the season it failed). A season
        the harness could not start is in ``not_started`` and in no count here.
        """
        seasons = self._arm_seasons(arm)
        failed = self.failed.get(arm, 0)
        n = len(seasons) + failed
        outcome: dict[str, bool] = {}
        for cls in self.soft_classes:
            k = failed + sum(1 for r in seasons if r.exceedances.get(cls, np.zeros(1)).max() > 0.0)
            outcome[cls] = passes_soft_gate(k, n, self.gate.epsilon, self.gate.delta)
        return outcome

    def banded_gate_outcome(
        self, arm: str, bands: dict[str, tuple[float, float]]
    ) -> dict[str, bool]:
        """Per soft class: does the arm clear the secondary (banded) gate?

        Same Clopper-Pearson test as :meth:`gate_outcome`, but a season counts as violating
        a class only if it stays more than ``depth`` outside its bound for more than
        ``min_hours`` consecutive hours, with ``bands[cls] = (depth, min_hours)``. A crashed
        season still counts as violating.
        """
        seasons = self._arm_seasons(arm)
        failed = self.failed.get(arm, 0)
        n = len(seasons) + failed
        outcome: dict[str, bool] = {}
        for cls in self.soft_classes:
            depth, min_hours = bands[cls]
            k = failed + sum(
                1
                for r in seasons
                if cls in r.exceedances
                and banded_violation(r.exceedances[cls], self.step_hours, depth, min_hours)
            )
            outcome[cls] = passes_soft_gate(k, n, self.gate.epsilon, self.gate.delta)
        return outcome

    def crop_gate_outcome(
        self, arm: str, step_seconds: int, rules: dict[str, dict[str, dict[str, Any]]]
    ) -> dict[str, bool] | None:
        """Per soft class: does the arm clear the gate under the crop rules?

        Same Clopper-Pearson test as :meth:`gate_outcome`; a season violates a class if
        its climate trace breaks any of ``rules[cls]`` (see
        :mod:`phytotron.metrics.crop_rules`). A crashed season still counts as violating.
        None if any completed season has no climate trace.
        """
        seasons = self._arm_seasons(arm)
        if any(r.climate is None for r in seasons):
            return None
        failed = self.failed.get(arm, 0)
        n = len(seasons) + failed
        outcome: dict[str, bool] = {}
        for cls in self.soft_classes:
            k = failed + sum(
                1 for r in seasons if _breaks_crop_rules(r, cls, step_seconds, rules[cls])
            )
            outcome[cls] = passes_soft_gate(k, n, self.gate.epsilon, self.gate.delta)
        return outcome

    def crop_gate_counts(
        self, arm: str, step_seconds: int, rules: dict[str, dict[str, dict[str, Any]]]
    ) -> dict[str, dict[str, float]] | None:
        """Per soft class: failing seasons (crashes included), seasons, and the one-sided
        Clopper-Pearson upper bound on the failing-season rate. None without traces."""
        seasons = self._arm_seasons(arm)
        if any(r.climate is None for r in seasons):
            return None
        failed = self.failed.get(arm, 0)
        n = len(seasons) + failed
        counts: dict[str, dict[str, float]] = {}
        for cls in self.soft_classes:
            k = failed + sum(
                1 for r in seasons if _breaks_crop_rules(r, cls, step_seconds, rules[cls])
            )
            bound = clopper_pearson_upper(k, n, self.gate.delta) if n else 1.0
            counts[cls] = {"failing": k, "seasons": n, "upper_bound": bound}
        return counts

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

    def online_cost(self, arm: str) -> float | None:
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
            and (not gate_passers_only or all(self.gate_outcome(a).values()))
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
        costs = {a: self.online_cost(a) for a in arms}
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
