# The economic MPC

Phytotron ships one model-based reference controller: `PhysicsOnlyMPC` in `phytotron/controllers/mpc.py`, an economic nonlinear model predictive controller on the GreenLight tomato model. It is the "physics only" arm of the benchmark: it plans with physics and has no learned part. This page explains what it does, how its solver is set up, and where each choice comes from. Validation results are on the [validation page](validation.md).

## What it does

Every control step (900 s in the benchmark), the controller:

1. reads the current greenhouse state and a weather forecast for the next hour;
2. finds the heating, CO2, ventilation, screen, and lamp plan over that hour that maximises fruit revenue minus heating, CO2, and electricity cost, with soft penalties for leaving the allowed CO2, temperature, and humidity band;
3. applies the first step of the plan and repeats.

The objective is the economic form of GreenLight-Gym2's `GreenhouseReward` (revenue minus operating cost minus penalties). Prices come from the `economic` block of `configs/benchmark.yaml`. The three penalty weights there (CO2 5e-5, temperature 5e-3, RH 7e-4) are the soft-constraint weights of Msaad et al. [msaad2025rlmpc], Table 2; temperature and RH equal GreenLight-Gym2's own reward weights, while its CO2 weight is 4e-4, so the MPC's objective is not GreenLight-Gym2's reward. The allowed climate band comes from its `constraints` block. Following van Laatum et al. [vanlaatum2026rlmpc], humidity is penalised from 2 percentage points below its ceiling, so the controller keeps a small margin.

The controller holds a **nominal** model only. In the benchmark this is the simulator's parameter set perturbed by the mismatch level (`phytotron.mismatch.perturb_params`), never the true parameters. `PhysicsController` enforces this. A controller that plans with the exact simulator parameters would be tested on its own answer key (an "inverse crime"), which makes physics-based controllers look better than they would be on a real greenhouse.

```python
from phytotron.controllers.base import NominalParameters
from phytotron.controllers.mpc import PhysicsOnlyMPC

mpc = PhysicsOnlyMPC(
    NominalParameters(nominal),  # perturbed parameters, never the plant's own
    horizon=4,  # 4 x 900 s = 1 hour ahead
    dt=900.0,
    objective="economic",
    economic=cfg.economic,  # prices from configs/benchmark.yaml
    climate_bounds=bounds,  # CO2, temperature, RH bands
)
```

`phytotron quickstart --controller mpc` runs it for a season.

## Formulation

- **Model.** The 28-state GreenLight ODE as implemented in GreenLight-Gym2 (`gl_gym.environments.models.ode`) [vanlaatum2024greenlight], based on GreenLight [katzin2020greenlight]. States are scaled per state (temperatures about 10, CO2 and vapour pressure about 1000, biomass about 100,000), because the raw states span about five orders of magnitude and the unscaled model is not solvable in an optimizer.
- **Transcription.** Direct collocation, Radau degree 2, one finite element per control interval, built with CasADi Opti [andersson2019casadi] and solved by IPOPT [wachter2006ipopt].
- **Horizon.** 4 intervals of 900 s, about one hour. Van Laatum et al. also use about one hour (12 intervals of 300 s), because longer horizons on this stiff model are computationally intractable.
- **Input rate limit.** `|u(t) - u(t-1)| <= 0.1 * u_max` per step, as in van Laatum et al.
- **Soft bounds.** L1 penalties through slack variables, each normalised to order 1 (CO2 by 1000 ppm, temperature by 10 C, RH by 100%) with the scale folded into the weight. Without this normalisation the dual variables spanned about seven orders of magnitude and IPOPT diverged.
- **Terminal cost (off by default).** `terminal_weight > 0` adds an L1 pull toward a target state at the horizon end. This is the hook RL-in-MPC uses. The physics-only arm runs with `terminal_weight=0`.

## Solver recipe

From van Laatum et al. [vanlaatum2026rlmpc]:

- **Quasi-Newton (L-BFGS) Hessian.** Avoids the exact second-order sensitivities that made an exact-Hessian solve fail.
- **Relaxed IPOPT tolerances** with acceptable-point termination, so IPOPT returns a good-enough point promptly.

Added so the solver recovers from bad steps:

- **A model that is finite everywhere.** IPOPT evaluates the model at trial points outside the feasible region during line search and restoration, and a NaN there is fatal. The ODE divides by CO2, vapour pressure, and live biomass, so these states are floored smoothly with `sqrt(z^2 + eps^2)` before every model call (`_domain_safe`). The saturation vapour pressure used by the humidity penalty gets a pole-safe denominator for the same reason.
- **A per-state box.** Temperatures are bounded to [-40, 80] C (never binding in practice, but it keeps iterates away from the region where the temperature-dependent terms blow up). All other states have a loose box, because a tight physical bound that dosing legitimately crosses makes the problem infeasible. The physical band is enforced softly by the penalties instead.
- **Warm control history.** The first step starts from heating and CO2 on, so the rate limit does not cripple heating at a cold season start.
- **Fallbacks.** If IPOPT stops without converging, its last iterate is used if it is finite and inside the control bounds ("salvage"). If not, the last control is held ("hold"). Every solve is counted as clean, salvage, or hold, and **solve success** = (clean + salvage) / total is reported per season.

Speed choices, each benchmarked on a cold-start closed loop (2018, day 59, horizon 4):

| Change | Effect |
|---|---|
| Build the NLP once, update only parameters (state, previous control, forecast) | The main win: the symbolic graph used to be rebuilt every step |
| Warm-start the controls from the previous solve (states are re-seeded from the current state) | Fewer IPOPT iterations |
| Cap iterations, short acceptable-point window | Bounds the worst-case step time |
| Flatten the graph from MX to SX (`expand=True`) | Fewer fallbacks at no fidelity cost |
| Radau degree 2 instead of 3 | About 2x faster; tracks the simulator to about 1% (largest 1.3%, vapour pressure, over 6 steps in `examples/validate_mpc.py`; air temperature 0.5%). Degree 1 is too coarse (about 10% on air temperature) |
| JIT compilation | Rejected: 30 s compile, no steady-state gain |

The result is a median solve of about 0.06 to 1 s per step depending on the season phase. In Growmatics' reference run of 24 July 2026 ("the study"; see docs/validation.md, section 1) the mean decision time was about 0.25 s per step, so a 60-day season (5760 steps) takes about 20 to 25 minutes.

## Linear solver and reproducibility

IPOPT's inner linear solver is MUMPS, which ships with the pip CasADi wheel. SPRAL is also bundled and slightly faster, but its parallel pivoting is not deterministic, so it is not used. The HSL solvers (MA27, MA57, MA86) [hsl] are listed by the build but not linked: they fail immediately and are caught by the fallback. `linear_solver` is a constructor argument, so an HSL-enabled CasADi build is a one-line change.

**Pin the math libraries to one thread.** With the default thread count, multithreaded sums round differently, and over thousands of solves that moves IPOPT onto a different path. The study's 2025 season gives -1.745 EUR/m2 with default threads and exactly the recorded -1.607615 with one thread. Call `phytotron.reproducibility.pin_threads()` before importing numpy; all examples do. The thread settings are written into every run record (`hardware_stamp`).

## Accuracy

There are three separate error sources:

1. **The simulator** (the benchmark's ground truth) integrates the ODE with CVODES. Integrating the same configuration twice gives identical states.
2. **The MPC's own discretisation** at zero mismatch. Against the simulator under the same controls, the error is below 0.03% on CO2, air temperature, vapour pressure, and fruit over 6 steps (`examples/validate_mpc.py`). Because the controller re-reads the true state every step, this error does not build up over a season.
3. **Model mismatch** (the perturbed nominal model). This error grows with the horizon in open loop, is corrected by feedback in closed loop, and is what the benchmark's mismatch ladder measures on purpose.

## Known behaviour

- **CasADi prints "NaN detected" warnings** during a season. These are trial points IPOPT rejects during its line search and then recovers from. They are not failures. The solve-success figure counts the real outcome.
- **The MPC is myopic.** One hour of foresight cannot value crop growth over weeks, so the physics-only arm is cautious with heating and CO2 and scores below the incumbent on net value. This is a property of the short horizon, documented in the study and in van Laatum et al., and the reason RL-in-MPC exists.
- **Control convention.** GreenLight-Gym2's `env.step` treats an action as an increment (for RL). The MPC outputs absolute controls, which must go through `env.step_raw_control`. The harness does this by reading `controller.control_kind`, and `tests/test_harness.py` checks it.

## References

- **[msaad2025rlmpc]** Msaad, S., Harraway, M., McAllister, R. D. (2025). RL-Guided MPC for Autonomous Greenhouse Control. arXiv:2506.13278. https://arxiv.org/abs/2506.13278
- **[vanlaatum2026rlmpc]** van Laatum, B., Msaad, S., van Henten, E. J., McAllister, R. D., Boersma, S. (2026). Improving greenhouse fruit-production control by integrating reinforcement learning into short-horizon model predictive control. arXiv:2607.07365. https://arxiv.org/abs/2607.07365
- **[vanlaatum2024greenlight]** van Laatum, B., van Henten, E. J., Boersma, S. (2025). GreenLight-Gym: Reinforcement learning benchmark environment for control of greenhouse production systems. IFAC-PapersOnLine 59(23), 437-442. https://doi.org/10.1016/j.ifacol.2025.11.827
- **[katzin2020greenlight]** Katzin, D., van Mourik, S., Kempkes, F., van Henten, E. J. (2020). GreenLight: An open source model for greenhouses with supplemental lighting: Evaluation of heat requirements under LED and HPS lamps. Biosystems Engineering 194, 61-81. https://doi.org/10.1016/j.biosystemseng.2020.03.010
- **[andersson2019casadi]** Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., Diehl, M. (2019). CasADi: a software framework for nonlinear optimization and optimal control. Mathematical Programming Computation 11(1), 1-36. https://doi.org/10.1007/s12532-018-0139-4
- **[wachter2006ipopt]** Wächter, A., Biegler, L. T. (2006). On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming. Mathematical Programming 106(1), 25-57. https://doi.org/10.1007/s10107-004-0559-y
- **[hsl]** STFC Rutherford Appleton Laboratory. HSL: A collection of Fortran codes for large scale scientific computation. https://www.hsl.rl.ac.uk/
