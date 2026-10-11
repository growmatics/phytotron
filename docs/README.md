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
