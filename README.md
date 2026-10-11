# Phytotron

**An open virtual phytotron for benchmarking greenhouse climate control.**

Developed and maintained by [Growmatics](https://growmatics.ai).

> Status: 0.1.0, the first public version (11 October 2026). The benchmark core and the reference controllers run and reproduce our recorded results exactly; see [Validation](docs/validation.md). Interfaces may change between minor versions until 1.0; every change is in the [changelog](CHANGELOG.md).

## What it is

Phytotron is a test bench for greenhouse climate controllers. It runs a controller through fixed, simulated growing seasons and scores it the same way every time: net economic value (fruit revenue minus heating, CO2 and electricity cost), fruit growth, decision time, and a crop-safety gate with rules from published sources. Any controller can plug in: a rule-based setpoint strategy, model predictive control, reinforcement learning, or a hybrid.

The goal is a neutral yardstick. Claims like "our controller saves 10% energy" become something anyone can rerun and check.

## Reference controller and scenario

- The **incumbent** is the reference baseline: a rule-based setpoint controller that behaves like a typical climate-computer configuration. Every result is reported against the incumbent.
- The **reference scenario** is a Venlo-type tomato greenhouse under Dutch weather.

A result reads like: "controller X beats the incumbent in the reference scenario by N% gas at equal yield, with no gate violations."

## Try it

```
pip install -e .
phytotron quickstart --controller incumbent
```

You also need the simulator's configuration files and weather data; [the quickstart](docs/quickstart.md) covers both in five steps. Installing needs `git` on the machine, because the simulator is pinned to a git tag.

## What is included

- A benchmark harness: fixed seasons, net economic value, safety profiles, a feasibility gate before any ranking, and Pareto ranking of value against decision time.
- A model-mismatch ladder: physics-based controllers plan with a deliberately perturbed model, never the simulator's own parameters.
- Reference controllers: the rule-based incumbent and an economic MPC ([how it works](docs/mpc.md)), plus a wrapper for reinforcement-learning policies (Phytotron ships no trained policy; the wrapper runs one you bring). Your own controller runs through the same grid with `phytotron run --controller name=module:Class` ([guide](docs/own-controller.md)).
- Weather tools: real historical weather from the Open-Meteo archive, in the simulator's format.

## Built on

Phytotron's current simulator back end is [GreenLight-Gym2](https://github.com/BartvLaatum/GreenLight-Gym2) (van Laatum et al.), which implements the [GreenLight](https://github.com/davkat1/GreenLight) greenhouse model (Katzin et al., Wageningen University & Research). We thank both teams for publishing their work openly.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE). The license follows from the GreenLight-Gym2 dependency.

## Security

Please report security problems privately, as described in [SECURITY.md](SECURITY.md).

## Disclaimer

Phytotron is simulation and benchmarking software. It is not a climate controller and must not drive greenhouse equipment.
