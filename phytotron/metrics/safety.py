"""Multi-measure safety profile per constraint class.

A single violation rate hides how failure is distributed, so satisfaction is
reported as a profile over per-season traces of one scalar quantity against a
``[lower, upper]`` bound: incidence, duration, cumulative degree-hours
(severity), worst excursion, and the season-level violation probability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SafetyProfile:
    """Per-class safety summary. Units follow the trace (e.g. degree-hours in K*h)."""

    incidence: float  # fraction of seasons with any violation
    mean_duration_h: float  # mean hours outside the bound per season
    degree_hours: float  # mean cumulative magnitude*hours outside the bound per season
    worst_excursion: float  # largest single-step magnitude outside the bound
    season_violation_prob: float  # season-level probability of any violation (= incidence)


def _exceedance(trace: NDArray[np.float64], lower: float, upper: float) -> NDArray[np.float64]:
    """Non-negative magnitude outside ``[lower, upper]`` at each step."""
    below = np.maximum(lower - trace, 0.0)
    above = np.maximum(trace - upper, 0.0)
    return below + above


def safety_profile_from_exceedance(
    season_exceedances: Sequence[NDArray[np.float64]],
    step_hours: float,
) -> SafetyProfile:
    """Safety profile from per-season exceedance series (magnitude outside bound, ``>= 0``).

    Use this when the environment already reports violation magnitudes (e.g. the
    GreenLight-Gym ``info`` fields ``temp_violation``, ``rh_violation``).
    """
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    if len(season_exceedances) == 0:
        raise ValueError("need at least one season")

    any_violation = np.zeros(len(season_exceedances), dtype=bool)
    durations = np.zeros(len(season_exceedances))
    degree_hours = np.zeros(len(season_exceedances))
    worst = 0.0
    for i, series in enumerate(season_exceedances):
        exc = np.abs(np.asarray(series, dtype=float))
        violating = exc > 0.0
        any_violation[i] = bool(violating.any())
        durations[i] = float(violating.sum()) * step_hours
        degree_hours[i] = float(exc.sum()) * step_hours
        worst = max(worst, float(exc.max(initial=0.0)))

    incidence = float(any_violation.mean())
    return SafetyProfile(
        incidence=incidence,
        mean_duration_h=float(durations.mean()),
        degree_hours=float(degree_hours.mean()),
        worst_excursion=worst,
        season_violation_prob=incidence,
    )


def banded_violation(
    exceedance: NDArray[np.float64], step_hours: float, depth: float, min_hours: float
) -> bool:
    """True if the exceedance stays above ``depth`` for more than ``min_hours`` in a row.

    The secondary (banded) gate: a brief, shallow excursion does not count, a sustained
    one does. ``depth`` is in the trace's unit (RH percentage points for humidity, C for
    temperature). "In a row" is literal: one step at or below ``depth`` ends the run.
    Durations are compared in whole seconds, so step sizes that do not divide an hour
    are not subject to floating-point rounding.
    """
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    step_seconds, min_seconds = round(step_hours * 3600), round(min_hours * 3600)
    run = longest = 0
    for over in np.asarray(exceedance, dtype=float) > depth:
        run = run + 1 if over else 0
        longest = max(longest, run)
    return longest * step_seconds > min_seconds


def safety_profile(
    season_traces: Sequence[NDArray[np.float64]],
    lower: float,
    upper: float,
    step_hours: float,
) -> SafetyProfile:
    """Compute the safety profile for one constraint over a set of seasons.

    Args:
        season_traces: one 1-D array of the controlled quantity per season.
        lower, upper: the constraint bounds.
        step_hours: hours represented by one timestep.
    """
    if lower > upper:
        raise ValueError(f"lower {lower} must not exceed upper {upper}")
    if len(season_traces) == 0:
        raise ValueError("need at least one season trace")
    exceedances = [_exceedance(np.asarray(t, dtype=float), lower, upper) for t in season_traces]
    return safety_profile_from_exceedance(exceedances, step_hours)
