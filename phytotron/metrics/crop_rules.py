"""Crop-safety rules: does one season's climate trace break a class's rule?

The season-level gate (:mod:`phytotron.metrics.gate`) needs one yes/no per season and
class. These rules decide it from the climate trace, in the forms the crop literature
uses (sources and numbers in ``configs/benchmark.yaml`` and ``docs/kpis.md``):

- ``rolling_mean``: the mean over a moving window leaves ``[low, high]`` (temperature
  integration: tomato responds to the 24 h and multi-day mean, not to short swings);
- ``light_mean``: the mean over a moving window of days leaves ``target +/- band``, where
  each day's target follows that day's outdoor radiation sum (needs the radiation trace);
- ``dose``: a dose of extreme values builds up inside a moving window. Each step beyond an
  onset adds its share of a full dose, with Körner and Challa's (2003) exponential weight:
  0 at the onset, 1 at the absolute limit (and capped at 1 beyond it, where the peak rule
  applies). The rule breaks when the dose in a window exceeds 1;
- ``peak``: any step outside ``[low, high]``;
- ``daily_hours``: the mean, over a moving window of days, of the daily hours above a
  threshold exceeds a limit (disease risk follows humid hours);
- ``longest_run``: one unbroken run above a threshold lasts longer than a limit.

A window longer than the trace is cut to the trace, so shortened smoke seasons still give
a value. Durations are counted in whole seconds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

RULE_PARAMS: dict[str, tuple[str, ...]] = {
    "rolling_mean": ("window_h",),
    "light_mean": ("base", "per_1000_j_cm2", "band", "window_days"),
    "dose": (
        "onset_low",
        "limit_low",
        "onset_high",
        "limit_high",
        "full_dose_min",
        "window_h",
        "heat_exponent",
        "cold_exponent",
    ),
    "peak": (),
    "daily_hours": ("threshold", "max_hours_per_day", "window_days"),
    "longest_run": ("threshold", "max_hours"),
}
_BOUNDED = ("rolling_mean", "peak")  # rules that need at least one of low / high


def _steps(hours: float, step_seconds: int) -> int:
    seconds = round(hours * 3600)
    if seconds % step_seconds:
        raise ValueError(f"{hours} h is not a whole number of {step_seconds} s steps")
    return seconds // step_seconds


def _window_sums(values: NDArray[np.float64], width: int) -> NDArray[np.float64]:
    """Sums over every full window of ``width`` (the whole trace if it is shorter)."""
    width = min(width, len(values))
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return cumulative[width:] - cumulative[:-width]


def _outside(values: NDArray[np.float64], low: float | None, high: float | None) -> bool:
    below = low is not None and bool((values < low).any())
    above = high is not None and bool((values > high).any())
    return below or above


def rolling_mean(
    trace: NDArray[np.float64],
    step_seconds: int,
    *,
    window_h: float,
    low: float | None = None,
    high: float | None = None,
) -> bool:
    width = _steps(window_h, step_seconds)
    means = _window_sums(trace, width) / min(width, len(trace))
    return _outside(means, low, high)


def light_mean(
    trace: NDArray[np.float64],
    step_seconds: int,
    *,
    radiation: NDArray[np.float64] | None = None,
    base: float,
    per_1000_j_cm2: float,
    band: float,
    window_days: int,
) -> bool:
    """Day means against light-following day targets, both averaged over ``window_days``.

    Each day's target is ``base + per_1000_j_cm2 * (radiation sum in J/cm2) / 1000``, from
    the outdoor global radiation (W/m2) at each step (Het Nieuwe Telen's rule of thumb).
    """
    if radiation is None:
        raise ValueError("light_mean needs the outdoor radiation trace")
    per_day = _steps(24.0, step_seconds)
    days = max(len(trace) // per_day, 1)
    usable = min(len(trace), days * per_day)
    temp = trace[:usable].reshape(days, -1).mean(axis=1)
    j_cm2 = np.asarray(radiation, dtype=float)[:usable].reshape(days, -1).sum(axis=1)
    target = base + per_1000_j_cm2 * j_cm2 * step_seconds / 1e4 / 1000.0
    width = min(int(window_days), days)
    gap = (_window_sums(temp, width) - _window_sums(target, width)) / width
    return bool((np.abs(gap) > band).any())


def dose_weights(
    trace: NDArray[np.float64],
    *,
    onset_low: float,
    limit_low: float,
    onset_high: float,
    limit_high: float,
    heat_exponent: float,
    cold_exponent: float,
) -> NDArray[np.float64]:
    """Per-step weight of extreme temperature, Körner and Challa (2003) Eq. 3 and 4.

    Heat: (T - onset) exp(heat_exponent (1/limit - 1/T)) / (limit - onset); cold:
    (onset - T) exp(-cold_exponent (1/limit - 1/T)) / (onset - limit), with T in C as in
    the paper. 0 inside the onsets, 1 at the limits, capped at 1 beyond them.
    """
    t_hot = np.clip(trace, onset_high, limit_high)
    heat = (
        (t_hot - onset_high)
        * np.exp(heat_exponent * (1.0 / limit_high - 1.0 / t_hot))
        / (limit_high - onset_high)
    )
    t_cold = np.clip(trace, limit_low, onset_low)
    cold = (
        (onset_low - t_cold)
        * np.exp(-cold_exponent * (1.0 / limit_low - 1.0 / t_cold))
        / (onset_low - limit_low)
    )
    return np.minimum(heat, 1.0) + np.minimum(cold, 1.0)


def dose(
    trace: NDArray[np.float64],
    step_seconds: int,
    *,
    onset_low: float,
    limit_low: float,
    onset_high: float,
    limit_high: float,
    full_dose_min: float,
    window_h: float,
    heat_exponent: float,
    cold_exponent: float,
) -> bool:
    weights = dose_weights(
        trace,
        onset_low=onset_low,
        limit_low=limit_low,
        onset_high=onset_high,
        limit_high=limit_high,
        heat_exponent=heat_exponent,
        cold_exponent=cold_exponent,
    )
    per_step = weights * (step_seconds / 60.0) / full_dose_min
    return bool((_window_sums(per_step, _steps(window_h, step_seconds)) > 1.0 + 1e-12).any())


def peak(
    trace: NDArray[np.float64],
    step_seconds: int,
    *,
    low: float | None = None,
    high: float | None = None,
    min_minutes: float = 0.0,
) -> bool:
    """Beyond a limit for ``min_minutes`` or more in a row (protocol 2; 0 = any step).

    Protocol 1 had no minimum: one step beyond a limit failed the season. Protocol 2 sets
    30 minutes, Körner and Challa's full-dose time, so a single 15-minute transient is
    judged by the dose rule instead.
    """
    if min_minutes <= 0:
        return _outside(trace, low, high)
    below = (trace < low) if low is not None else np.zeros(len(trace), dtype=bool)
    above = (trace > high) if high is not None else np.zeros(len(trace), dtype=bool)
    run = longest = 0
    for out in below | above:
        run = run + 1 if out else 0
        longest = max(longest, run)
    return longest * step_seconds >= round(min_minutes * 60)


def daily_hours(
    trace: NDArray[np.float64],
    step_seconds: int,
    *,
    threshold: float,
    max_hours_per_day: float,
    window_days: int,
) -> bool:
    per_day = _steps(24.0, step_seconds)
    days = max(len(trace) // per_day, 1)
    over = (trace[: days * per_day] > threshold).astype(float)
    hours = over.reshape(days, -1).sum(axis=1) * step_seconds / 3600.0
    width = min(int(window_days), days)
    return bool((_window_sums(hours, width) / width > max_hours_per_day).any())


def longest_run(
    trace: NDArray[np.float64],
    step_seconds: int,
    *,
    threshold: float,
    max_hours: float,
) -> bool:
    run = longest = 0
    for over in trace > threshold:
        run = run + 1 if over else 0
        longest = max(longest, run)
    return longest * step_seconds > round(max_hours * 3600)


RULES: dict[str, Callable[..., bool]] = {
    "rolling_mean": rolling_mean,
    "light_mean": light_mean,
    "dose": dose,
    "peak": peak,
    "daily_hours": daily_hours,
    "longest_run": longest_run,
}


def check_rule(spec: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if a rule spec has an unknown kind or misses a parameter."""
    kind = spec.get("kind")
    if kind not in RULES:
        raise ValueError(f"rule kind must be one of {list(RULES)}, got {kind!r}")
    missing = [p for p in RULE_PARAMS[kind] if p not in spec]
    if missing:
        raise ValueError(f"{kind} rule misses {missing}")
    if kind in _BOUNDED and "low" not in spec and "high" not in spec:
        raise ValueError(f"{kind} rule needs low, high or both")


def broken_rules(
    trace: NDArray[np.float64],
    step_seconds: int,
    rules: Mapping[str, Mapping[str, Any]],
    radiation: NDArray[np.float64] | None = None,
) -> list[str]:
    """Names of the rules this season's trace breaks, in config order.

    ``radiation`` is the outdoor global radiation per step (W/m2), for ``light_mean``.
    """
    values = np.asarray(trace, dtype=float)
    if values.size == 0:
        raise ValueError("empty trace")
    broken = []
    for name, spec in rules.items():
        params = {k: v for k, v in spec.items() if k != "kind"}
        if spec["kind"] == "light_mean":
            params["radiation"] = radiation
        if RULES[spec["kind"]](values, step_seconds, **params):
            broken.append(name)
    return broken


def humidity_deficit(temp_c: NDArray[np.float64], rh: NDArray[np.float64]) -> NDArray[np.float64]:
    """Humidity deficit in g/m3: how much more water vapour the air could hold.

    Saturation vapour pressure by the Tetens formula, converted to vapour density with
    the ideal gas law.
    """
    t = np.asarray(temp_c, dtype=float)
    saturation_pa = 610.78 * np.exp(17.27 * t / (t + 237.3))
    saturation_g_m3 = saturation_pa * 18.015 / (8.314 * (t + 273.15))
    return saturation_g_m3 * (1.0 - np.asarray(rh, dtype=float) / 100.0)
