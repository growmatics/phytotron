"""GreenLight-Gym2 v0.2.1 resamples the season's weather on a grid whose spacing is not
exactly the control step and depends on the season length (``docs/validation.md``,
"Weather grid"). This test pins that behaviour for the shipped pin, so a pin move that
fixes it is noticed and the documentation updated."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

gl_utils = pytest.importorskip("gl_gym.environments.utils")

RAW_DT = 300  # s, the 5-minute spacing of the weather files
STEP = 900  # s, the control step
DAY = 86400


def _write_year(folder: Path, days: int) -> None:
    """A synthetic weather year whose air temperature equals the time in seconds."""
    t = np.arange(0, days * DAY, RAW_DT)
    folder.mkdir(parents=True)
    header = "time,global radiation,air temperature,RH,wind speed,sky temperature\n"
    rows = (f"{x},100,{x},70,2,5\n" for x in t)
    (folder / "2024.csv").write_text(header + "".join(rows))


@pytest.mark.parametrize(("days", "drift_end_s"), [(1, 452.4), (3, 676.8)])
def test_resampled_grid_spacing_depends_on_the_season_length(
    tmp_path: Path, days: int, drift_end_s: float
) -> None:
    _write_year(tmp_path / "Synthetic", days + 2)
    horizon_days = 1
    data = gl_utils.load_weather_data(
        str(tmp_path), "Synthetic", 2024, 0, days, horizon_days, STEP, 10
    )
    times = data[:, 1]  # air temperature column carries the time
    spacing = np.diff(times)
    # Expected from the loader's linspace over Ns + Np raw samples, resampled to ns points.
    ns_raw = int(np.ceil(days * DAY / RAW_DT)) + int(np.ceil(horizon_days * DAY / RAW_DT)) + 1
    ns = int((RAW_DT / STEP) * ns_raw)
    expected = (ns_raw - 1) * RAW_DT / (ns - 1)
    assert spacing == pytest.approx(expected, abs=1e-6)
    assert expected != STEP  # the grid is not the control step
    # The read-out time drifts from the true step time; at the end of the season by:
    assert (expected - STEP) * (days * DAY / STEP) == pytest.approx(drift_end_s, abs=1.0)
