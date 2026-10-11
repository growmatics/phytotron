"""Tests for the crop-safety rules of the gate (pure, no simulator)."""

import numpy as np
import pytest

from phytotron.metrics.crop_rules import (
    broken_rules,
    check_rule,
    daily_hours,
    dose,
    dose_weights,
    humidity_deficit,
    light_mean,
    longest_run,
    peak,
    rolling_mean,
)

STEP = 900  # s, the benchmark's control step
PER_HOUR = 4
PER_DAY = 96


def test_rolling_mean_uses_the_window_not_single_steps() -> None:
    # 18 C with one 2 h swing to 30 C: the 24 h mean is 19 C, inside 14-22.
    trace = np.full(3 * PER_DAY, 18.0)
    trace[100:108] = 30.0
    assert not rolling_mean(trace, STEP, window_h=24, low=14.0, high=22.0)
    # A whole warm day moves the 24 h mean to 24 C.
    trace[PER_DAY : 2 * PER_DAY] = 24.0
    assert rolling_mean(trace, STEP, window_h=24, high=22.0)
    assert not rolling_mean(trace, STEP, window_h=24, low=14.0)


def test_rolling_mean_window_longer_than_trace_uses_the_whole_trace() -> None:
    assert rolling_mean(np.full(10, 25.0), STEP, window_h=144, high=22.0)
    assert not rolling_mean(np.full(10, 20.0), STEP, window_h=144, high=22.0)


def test_rolling_mean_rejects_windows_that_are_not_whole_steps() -> None:
    with pytest.raises(ValueError, match="whole number"):
        rolling_mean(np.zeros(10), STEP, window_h=0.1, high=1.0)


DOSE = {
    "onset_low": 14.0,
    "limit_low": 10.0,
    "onset_high": 30.0,
    "limit_high": 34.0,
    "full_dose_min": 30.0,
    "window_h": 24,
    "heat_exponent": 160.0,
    "cold_exponent": 30.0,
}
WEIGHTS = {k: v for k, v in DOSE.items() if k not in ("full_dose_min", "window_h")}


def test_dose_weights_follow_koerner_and_challa() -> None:
    temps = np.array([20.0, 31.0, 32.0, 33.0, 34.0, 36.0, 13.0, 12.0, 11.0, 10.0, 5.0])
    expected = [0.0, 0.159, 0.373, 0.650, 1.0, 1.0, 0.125, 0.303, 0.571, 1.0, 1.0]
    np.testing.assert_allclose(dose_weights(temps, **WEIGHTS), expected, atol=1e-3)


def test_dose_fills_after_30_minutes_at_a_limit() -> None:
    trace = np.full(PER_DAY, 20.0)
    trace[:2] = 34.0  # 30 min at the heat limit: exactly a full dose, not more
    assert not dose(trace, STEP, **DOSE)
    trace[2] = 32.0  # 15 more minutes at weight 0.37 inside the same 24 h
    assert dose(trace, STEP, **DOSE)


def test_dose_counts_cold_and_heat_and_resets_outside_the_window() -> None:
    trace = np.full(3 * PER_DAY, 20.0)
    trace[0] = 10.0  # 15 min at the cold limit: half a dose
    trace[1] = 34.0  # 15 min at the heat limit: the other half
    assert not dose(trace, STEP, **DOSE)
    trace[2] = 34.0
    assert dose(trace, STEP, **DOSE)
    later = np.full(3 * PER_DAY, 20.0)
    later[0:2] = 10.0
    later[PER_DAY + 10] = 34.0  # more than 24 h later: separate windows
    assert not dose(later, STEP, **DOSE)


LIGHT = {"base": 17.5, "per_1000_j_cm2": 1.5, "band": 4.0, "window_days": 6}


def _sunny_days(days: int, j_cm2: float) -> np.ndarray:
    """Radiation per step giving ``j_cm2`` per day, all of it in the first half day."""
    day = np.zeros(PER_DAY)
    day[: PER_DAY // 2] = j_cm2 * 1e4 / (PER_DAY // 2 * STEP)
    return np.tile(day, days)


def test_light_mean_target_follows_daily_radiation() -> None:
    sun = _sunny_days(6, 2000.0)  # target 17.5 + 3.0 = 20.5 C, band 16.5 to 24.5
    assert not light_mean(np.full(6 * PER_DAY, 20.0), STEP, radiation=sun, **LIGHT)
    assert light_mean(np.full(6 * PER_DAY, 16.0), STEP, radiation=sun, **LIGHT)
    dark = _sunny_days(6, 0.0)  # target 17.5 C, band 13.5 to 21.5
    assert not light_mean(np.full(6 * PER_DAY, 16.0), STEP, radiation=dark, **LIGHT)
    assert light_mean(np.full(6 * PER_DAY, 22.0), STEP, radiation=dark, **LIGHT)


def test_light_mean_needs_radiation() -> None:
    with pytest.raises(ValueError, match="radiation"):
        light_mean(np.full(PER_DAY, 20.0), STEP, **LIGHT)
    rules = {"m": {"kind": "light_mean", **LIGHT}}
    trace = np.full(PER_DAY, 30.0)
    assert broken_rules(trace, STEP, rules, radiation=np.zeros(PER_DAY)) == ["m"]


def test_humidity_deficit() -> None:
    # At 18 C, 87 % RH leaves about 2 g/m3; saturated air leaves none.
    deficit = humidity_deficit(np.array([18.0, 18.0]), np.array([87.0, 100.0]))
    np.testing.assert_allclose(deficit, [2.0, 0.0], atol=0.05)


def test_peak() -> None:
    assert peak(np.array([20.0, 34.5]), STEP, low=10.0, high=34.0)
    assert peak(np.array([9.9, 20.0]), STEP, low=10.0)
    assert not peak(np.array([10.0, 34.0]), STEP, low=10.0, high=34.0)


def test_peak_with_a_minimum_duration() -> None:
    """Protocol 2: one 15-minute step beyond a limit is not a peak failure; two in a row are."""
    one_step = np.array([20.0, 9.9, 20.0, 20.0, 35.0, 20.0])
    assert peak(one_step, STEP, low=10.0, high=34.0)  # protocol 1: any step
    assert not peak(one_step, STEP, low=10.0, high=34.0, min_minutes=30.0)
    assert peak(np.array([20.0, 9.9, 9.8, 20.0]), STEP, low=10.0, high=34.0, min_minutes=30.0)
    assert peak(np.array([20.0, 34.5, 34.1]), STEP, high=34.0, min_minutes=30.0)
    # Two separate single steps do not add up: the minimum is consecutive.
    assert not peak(np.array([9.9, 20.0, 9.9, 20.0]), STEP, low=10.0, min_minutes=30.0)
    # Below and above in a row count as one run beyond the limits.
    assert peak(np.array([9.9, 35.0]), STEP, low=10.0, high=34.0, min_minutes=30.0)


def test_daily_hours_takes_the_mean_over_days() -> None:
    days = np.full((7, PER_DAY), 80.0)
    days[0, : 10 * PER_HOUR] = 95.0  # one day with 10 humid hours, a mean of 10/7 h/day
    trace = days.ravel()
    assert not daily_hours(trace, STEP, threshold=90.0, max_hours_per_day=7.0, window_days=7)
    # Humid hours spread over a week count: dry breaks do not reset them.
    days[:, : 8 * PER_HOUR] = 95.0  # 8 h every day
    assert daily_hours(days.ravel(), STEP, threshold=90.0, max_hours_per_day=7.0, window_days=7)


def test_daily_hours_on_a_short_season() -> None:
    trace = np.full(PER_DAY // 2, 95.0)  # half a day, all humid: 12 h
    assert daily_hours(trace, STEP, threshold=90.0, max_hours_per_day=7.0, window_days=7)


def test_longest_run_needs_an_unbroken_run() -> None:
    run = np.full(49 * PER_HOUR, 95.0)
    assert longest_run(run, STEP, threshold=93.0, max_hours=48.0)
    assert not longest_run(run[: 48 * PER_HOUR], STEP, threshold=93.0, max_hours=48.0)
    broken = run.copy()
    broken[24 * PER_HOUR] = 90.0
    assert not longest_run(broken, STEP, threshold=93.0, max_hours=48.0)


def test_broken_rules_lists_names_in_order() -> None:
    rules = {
        "peak": {"kind": "peak", "low": 10.0, "high": 34.0},
        "mean_24h": {"kind": "rolling_mean", "window_h": 24, "high": 26.0},
    }
    assert broken_rules(np.full(PER_DAY, 35.0), STEP, rules) == ["peak", "mean_24h"]
    assert broken_rules(np.full(PER_DAY, 20.0), STEP, rules) == []
    with pytest.raises(ValueError, match="empty"):
        broken_rules(np.array([]), STEP, rules)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"kind": "average"}, "kind"),
        ({"kind": "longest_run", "threshold": 93.0}, "max_hours"),
        ({"kind": "peak"}, "low, high"),
    ],
)
def test_check_rule_rejects_bad_specs(spec: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        check_rule(spec)
