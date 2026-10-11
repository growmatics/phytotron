# Quickstart

Run the rule-based incumbent through one simulated season and read its scores. About ten minutes from a clean machine, most of it installing.

## 1. Install

Phytotron needs Python 3.11 or newer.

```
git clone https://github.com/growmatics/phytotron.git
cd phytotron
python -m venv .venv
.venv/bin/pip install -e .
```

This installs GreenLight-Gym2 v0.2.1, the simulator, from GitHub.

## 2. Scenario files

Phytotron ships the two GreenLight-Gym2 configuration files the benchmark reads (the scenario and the incumbent's parameters), as verbatim copies with their hashes (`phytotron/scenarios/`). Nothing to do. To run against a GreenLight-Gym2 checkout instead, set `GLG2_CONFIG_DIR=<checkout>/gl_gym/configs/envs`.

## 3. Fetch weather

The reference scenario uses real historical weather for Amsterdam from the free Open-Meteo archive (no key needed):

```
.venv/bin/phytotron fetch-weather --lat 52.37 --lon 4.90 --years 2025 \
    --location Amsterdam_OM --weather-dir weather
export GLG2_WEATHER_DIR=$PWD/weather
```

Use `--years 2014-2025` for all benchmark years (8 training, 4 held out).

## 4. Run

Installing the package gives a `phytotron` command with `quickstart`, `run` and `fetch-weather`; each takes `--help`. The scripts in `examples/` do the same and stay for the documented commands.

```
.venv/bin/phytotron quickstart --controller incumbent --year 2025
```

The output looks like this:

```
incumbent, simulated season, Amsterdam_OM 2025, 60 days
  net economic value: +3.088 EUR/m2
    = revenue 17.687 - heating 7.623 - CO2 2.070 - electricity 4.906
  fruit growth:       11.05 kg/m2 fresh
  decision time:      0.1 ms/step
  outside the allowed band (hours; size x hours; worst step):
    temperature:   43.0 h     51.7 C.h  worst 5.69 C
            co2:    0.0 h      0.0 ppm.h  worst 0.00 ppm
       humidity:  448.8 h    824.7 %RH.h  worst 10.42 %RH
           lamp:    0.0 h      0.0 on.h  worst 0.00 on
```

- **Net economic value**: fruit revenue minus heating, CO2, and electricity cost over the season, per square metre, with each part shown. Fixed costs are left out because they do not depend on the controller.
- **Fruit growth**: revenue divided by the fruit price, the growth of fruit fresh weight on the plants over the season. It is not harvested yield.
- **Outside the allowed band**: per climate class, the hours spent outside the band in `configs/benchmark.yaml` (CO2 300 to 1600 ppm, air 15 to 34 C, RH 50 to 85%), the size of the violation times its duration, and the worst single step.

The net value should match the 2025 row on the [validation page](validation.md). On 2026-10-09 a fresh download of 2025 was byte-identical to the file behind Growmatics' reference run of 24 July 2026 ("the study"; see docs/validation.md, section 1). If the archive revises its data later, the number can shift slightly.

## 5. Try the economic MPC

```
.venv/bin/phytotron quickstart --controller mpc
.venv/bin/phytotron quickstart --controller mpc --mismatch 0.10
```

The MPC runs 3 days by default, about a minute. The second run gives the MPC a model with 10% parameter error, the kind of gap a real greenhouse would have. `--days 60` runs the full season, about 20 to 25 minutes. CasADi prints "NaN detected" warnings along the way. They are rejected solver trial points, not failures; see [the MPC page](mpc.md#known-behaviour).

## Short seasons are not comparable to full ones

A shortened season (`--days` below 60) is not the first days of the full season: the shipped simulator resamples the weather on a grid that depends on the season length, so the same start day reads slightly different weather in a 12-day and a 60-day run ([validation, section 5](validation.md#5-weather-grid-of-the-shipped-simulator)). Use short runs to check that things work; compare controllers on full seasons only.

## Reproducibility

Each script pins the math libraries to one thread before NumPy loads. MPC results depend on it. If you write your own script, start it with:

```python
from phytotron.reproducibility import pin_threads

pin_threads()
```
