# Docs

- [Quickstart](quickstart.md): install, fetch weather, run the incumbent and the economic MPC.
- [The economic MPC](mpc.md): what the reference MPC does, its solver setup, and where each choice comes from.
- [Validation](validation.md): what has been checked, with the numbers, including what does not match.
- [KPIs and the feasibility gate](kpis.md): how a controller is scored, and why the gate comes before ranking.
- [Writing your own controller](own-controller.md): the interface, the controls, and a template.
- [Results](results.md): the reference controllers over the held-out seasons and the mismatch ladder, under the crop-safety rules.

## Glossary

- **Inverse crime.** Testing a controller against a simulator that is identical to the controller's own model. The controller is then graded on its own answer key, which makes model-based methods look better than they would on a real greenhouse. The term comes from inverse problems. Phytotron prevents it: a model-based controller gets a nominal model, the simulator's parameters perturbed by the mismatch level (`phytotron.mismatch.perturb_params`), never the simulator's own parameters. Mismatch 0.0 is reported only as a labelled inverse-crime reference.
- **Arm.** One controller type in the benchmark comparison. The code numbers them:
  - Arm 1: the incumbent, a rule-based setpoint controller (`RuleBasedIncumbent`).
  - Arm 2: black-box RL, a trained policy run through the shared harness (`RLController`). Phytotron ships no trained policy: the wrapper runs one you bring, and no RL result appears on the results page.
  - Arm 3: physics-only economic MPC (`PhysicsOnlyMPC`).
  - Arm 4: RL-in-MPC, the economic MPC with a terminal cost learned by RL. Planned, not yet in this repo.
- **Registered arm.** A controller you bring, registered on the command line with `--controller name=module:attr`, scored and gated like the built-in arms. The run record lists them under its `custom_arms` key, which keeps that name.
- **Season.** One simulated crop season: a weather year, a mismatch level and a seed. The unit the gate counts. Not "episode", "condition" or "run".
- **Mismatch level.** The size of the model error given to a model-based arm, from the config's `sweep_magnitudes`. Mismatch 0 is the inverse-crime reference.
- **Strict, banded and crop gate.** The three per-season rules, all reported: outside the band at any step (strict); more than a depth outside for more than an hour (banded); any crop rule of the class broken (crop, the primary gate). Not "hard gate" or "soft gate".
- **Soft class, reported class.** The config's `kind` of a climate class: `soft` is gated, `reported` is profiled only. Prose may say "the gated classes".
- **Verdict.** An arm's pooled gate result: for a model-based arm over its seasons with mismatch above 0, for a model-free arm over all its seasons. Each mismatch level is also gated on its own.
- **Protocol version.** The number raised with every change to a crop rule, printed on every result and stored in every summary. Not "rules version".
- **Decision time.** The mean wall-clock seconds per control step, the second Pareto axis. The summary key is `online_cost_seconds`, which keeps that name.
- **Run record, season record.** `run_record.json`: what a run was launched with (commit, config hash, input hashes, arm settings). A season record: one file in `seasons/`, a completed season or a failure with its stage.
- **Simulator.** GreenLight-Gym2 as the plant the controllers act on. Not "testbed" (the config key `testbed` keeps its name), not "environment" in prose (the identifier `env` stays), never "digital twin".

Keys and flags that keep an older word because they cannot change: `custom_arms`, `--controller`, `seeds_per_condition`, `testbed`, `kind: soft`, `online_cost_seconds`.
