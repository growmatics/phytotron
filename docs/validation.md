# Validation

What has been checked, how, and with what result. Every result on this page is simulated: it comes from the GreenLight-Gym2 simulator, not from a real greenhouse. Every number comes from a script in this repo that anyone can rerun. Where a check does not match, this page says so.

All runs: GreenLight-Gym2 v0.2.1 (commit `8bb5b28`), CasADi 3.6.7, NumPy 1.26.4, Python 3.12, math libraries pinned to one thread (`phytotron.reproducibility.pin_threads`). Run date 2026-10-09.

## Summary

| Check | Script | Result |
|---|---|---|
| Unit tests: gate, safety metrics, Pareto, mismatch, harness, MPC math | `pytest` | all tests pass; coverage above the 90% floor |
| Incumbent reproduces the study's benchmark run, 4 held-out seasons | `examples/reproduce_study_incumbent.py` | Exact on all 4 |
| Economic MPC reproduces the study's benchmark run, one season | `examples/reproduce_study_mpc.py` | Exact, with one math thread |
| MPC prediction model vs the simulator (L2) | `examples/validate_mpc.py` | Within 1.3% over 6 steps (vapour pressure; air temperature 0.5%) |
| MPC solver status (L3) | `examples/validate_mpc.py` | 4 of 4 steps solved, controls in bounds |
| MPC vs van Laatum et al.'s published standalone MPC (L5) | `examples/l5_reproduce.py` | Same penalty pattern; economic score higher than published |

## 1. Reproduction of the study's benchmark run

Phytotron's benchmark code was extracted from the code behind Growmatics' forthcoming benchmark study of physics-informed greenhouse control on GreenLight-Gym2. Its recorded run of 24 July 2026 is the reference below ("the study"). These checks prove the extraction changed nothing: the same controller on the same season gives the same number as the study's recorded run.

**Incumbent (rule-based), held-out years, Amsterdam, start day 59, 60 days, seed 0.** Simulated net economic value in EUR/m2:

| Year | Phytotron | The study | Difference |
|---|---|---|---|
| 2022 | +2.044326 | +2.044326 | 0.0 |
| 2023 | -9.444211 | -9.444211 | 0.0 |
| 2024 | -6.654230 | -6.654230 | 0.0 |
| 2025 | +3.087514 | +3.087514 | 0.0 |

**Economic MPC, 2025, zero model mismatch, seed 0.**

| Run | Simulated net value (EUR/m2) | Solve success |
|---|---|---|
| The study | -1.607615 | 1.0 |
| Phytotron, one math thread | -1.607615 (difference 0.0) | 1.0 |
| Phytotron, default threads | -1.745471 | 0.99983 |

The default-thread run differs because multithreaded sums round differently, and over 5,760 solves that sends the solver down a different path. The study's run used one thread. **MPC numbers are only reproducible with the threads pinned**, so every Phytotron script pins them and the run record stores the setting. See [the MPC page](mpc.md#linear-solver-and-reproducibility).

## 2. MPC validation layers

The economic MPC is validated in layers, from its math up to published results.

- **L1, the math.** Collocation coefficients, the scaled state box, the domain-safe flooring, and the rule that the MPC refuses the simulator's true parameters. Unit tests in `tests/test_mpc.py`, run in CI.
- **L2, prediction-model fidelity.** The MPC's own one-interval model, the one its optimisation problem plans with (one Radau element of degree 2 per 900 s step, the scaled and domain-safe ODE, the nominal parameters), and the simulator's CVODES integration are driven from the same state with the same fixed controls for 6 steps (Amsterdam 2018, day 59). Largest relative error: CO2 0.02%, air temperature 0.51%, vapour pressure 1.33%, fruit dry matter below 0.01%. Until 11 October 2026 the script measured a finer integrator (ten elements of degree 3) that the controller never uses; it reported errors of 0.02% and less, which was the integrator's accuracy, not the controller's model.
- **L3, solver status.** Four consecutive closed-loop steps from a cold season start, with the benchmark's controller (economic objective, 4-step horizon, built through the same factory as a benchmark season): 4 of 4 solved, all controls inside [0, 1]. Until 11 October 2026 the script built a different controller (placeholder objective, 3-step horizon).
- **L5, published numbers.** See below. (L4, a behavioural check over a full season, is covered by the reproduction of the study's run in section 1.)

## 3. Comparison with van Laatum et al. (L5)

Van Laatum et al. [vanlaatum2026rlmpc] report a standalone economic MPC on GreenLight-Gym2 for one day: Bleiswijk, 1 June 2023, 300 s steps, a 12-step (1 hour) horizon, 288 steps. `examples/l5_reproduce.py` runs Phytotron's MPC at their configuration (the `l5_reproduction` block of `configs/benchmark.yaml`) and scores it with GreenLight-Gym2's own reward.

| Quantity (simulated, one day) | Phytotron | Van Laatum et al. |
|---|---|---|
| Economic performance (EPI), EUR/m2 | +0.129 | -0.004 |
| Penalty | 0.026 | 0.024 |
| Objective (EPI minus penalty) | +0.102 | -0.028 |
| Penalty from temperature / CO2 / humidity | 0.000 / 0.000 / 0.026 | mostly humidity |
| Fruit revenue, EUR/m2 | 0.228 | about 0.05 |
| Median solve time | 0.12 s | not reported |

**Reading.**

- An exact match is not expected. Van Laatum et al. integrate the model with CVODES inside the optimizer and note that solutions are locally optimal up to tolerance. Phytotron uses direct collocation, a different transcription of the same model.
- **The penalty matches** within 10%, and it comes entirely from humidity with no temperature or CO2 violations. That is the same pattern they report.
- **The economic score is higher**: +0.13 against -0.004. Phytotron's MPC earns more fruit revenue on this day than theirs, for similar costs. A higher score than the reference is not by itself an error, but it is unexplained, and we do not claim it as an improvement.
- The study's first L5 run (July 2026) gave a penalty of 0.024 and an EPI of +0.10. The study's MPC code changed after that run (work on solver failures), and the current code gives the values above. Phytotron's MPC code is identical to the study's current code, so this is the controller's current behaviour, not an extraction error.

**Status:** the controller reproduces the reference's penalty and qualitative behaviour, not its economic score. Closing the gap would need their exact solver setup (CVODES in the loop), which is a possible later check.

## 4. What is not validated

- **Against real greenhouses.** Phytotron is checked against its own reference runs and one published simulation study. It is not yet checked against measured greenhouse data. GreenLight itself was evaluated against measured data from greenhouse trials [katzin2020greenlight]. A calibration against the Autonomous Greenhouse Challenge datasets is planned.
- **One simulator.** All results come from GreenLight-Gym2. Another implementation of the GreenLight model would not give identical numbers.
- **One crop and one season shape.** Tomato, a 60-day spring season in Amsterdam weather.
- **The strict humidity gate saturates.** In the study, every season of every controller exceeds the 85% RH ceiling at some step, so the strict any-step gate passes no one. The banded gate (more than 2 points above the ceiling for more than one hour) separates the controllers' violation rates, from 12 of 20 seasons to 120 of 120, but at the 5 % tolerance it passes no one either. In Phytotron's own full run (2026-10-10) every MPC season and every incumbent season violated it. Both gates are reported.

## 5. Weather grid of the shipped simulator

Found on 2026-10-10: the same start day gives a slightly different season when the season length differs (incumbent, 2024 weather: minimum air temperature 8.74 C in a 12-day run and 8.98 C in a 60-day run). The cause is in GreenLight-Gym2 v0.2.1's weather loader (`gl_gym/environments/utils.py`, `load_weather_data`). It cuts the 5-minute weather to the season plus a 49-day horizon and resamples it with `np.linspace` to a point count rounded from that length, so the resampled spacing is not exactly the 900-second control step, and it depends on the season length:

| Season | Resampled spacing | Weather read-out drift by the season's end |
|---|---|---|
| 60 days (the benchmark) | 900.086 s | 8.3 minutes late |
| 12 days | 900.154 s | 3.0 minutes late |
| 1 day (smoke run) | 904.7 s | 7.5 minutes late |

The outdoor temperature a step reads differs between the 12- and 60-day grids by up to 0.035 C and the radiation by up to 5.4 W/m2 over the first 12 days, enough to move a season's minimum temperature by a quarter of a degree. Consequences for Phytotron:

- Benchmark numbers come from full 60-day seasons only. Smoke and shortened runs are not comparable to them, and the quickstart says so.
- Every controller sees the same grid for the same season length, so comparisons within the benchmark are fair.
- The grid is part of the shipped simulator pin and is kept as is, so Phytotron still reproduces the study's recorded seasons exactly and matches upstream GreenLight-Gym2. `tests/test_weather_grid.py` pins the behaviour; when a pin move fixes it, the test fails and this section and the recorded results are updated under a new protocol version.

## References

- **[vanlaatum2026rlmpc]** van Laatum, B., Msaad, S., van Henten, E. J., McAllister, R. D., Boersma, S. (2026). Improving greenhouse fruit-production control by integrating reinforcement learning into short-horizon model predictive control. arXiv:2607.07365. https://arxiv.org/abs/2607.07365
- **[katzin2020greenlight]** Katzin, D., van Mourik, S., Kempkes, F., van Henten, E. J. (2020). GreenLight: An open source model for greenhouses with supplemental lighting: Evaluation of heat requirements under LED and HPS lamps. Biosystems Engineering 194, 61-81. https://doi.org/10.1016/j.biosystemseng.2020.03.010
