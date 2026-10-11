# Writing your own controller

Any controller that implements two methods can run in Phytotron and be scored like the reference controllers. The template is [`examples/my_controller.py`](../examples/my_controller.py), a 20-line thermostat.

## The interface

```python
class Controller(Protocol):
    def reset(self, seed: int) -> None: ...
    def act(self, ctx: ControlContext) -> NDArray[np.float64]: ...
```

`reset` is called once at the start of each season. Clear any internal state there, and seed any randomness from `seed`, so the same season gives the same result.

`act` is called every control step (900 s) and returns the six controls. It receives a `ControlContext`:

| Field | What it is | Who uses it |
|---|---|---|
| `obs` | the environment's observation vector (22 entries) | learned policies |
| `state` | the simulator's full state (28 entries) | model-based controllers |
| `forecast` | the weather for the next 24 steps (one row of 10 values per step) | model-based controllers |
| `timestep` | step index within the season | anything time-dependent |
| `env` | the live GreenLight-Gym2 environment | controllers that need its internals |

Read what you need and ignore the rest. Useful state entries: index 0 is CO2, 2 is air temperature in C, 15 is vapour pressure, 25 is fruit dry matter. The full list is in GreenLight-Gym2's model definition.

## The controls

Six values, each in [0, 1]:

| Index | Control |
|---|---|
| 0 | heating |
| 1 | CO2 dosing |
| 2 | thermal screen |
| 3 | ventilation |
| 4 | lamps |
| 5 | blackout screen |

Set `control_kind` on your class to say what `act` returns:

- `control_kind = "absolute"` (the default): the controls themselves. The rule-based incumbent, the MPC and the example thermostat do this.
- `control_kind = "incremental"`: changes in [-1, 1], which the simulator adds to the current controls at up to 0.1 per step. Reinforcement-learning policies trained on GreenLight-Gym2 usually do this; wrap one in `phytotron.controllers.rl.RLController`.

Getting this wrong does not raise an error. The plant just receives different controls than you planned, so check it first if results look strange.

## Run one season

```
GLG2_WEATHER_DIR=weather python examples/my_controller.py --days 60
```

```
thermostat, simulated season, Amsterdam_OM 2025, 60 days
  net economic value: +4.809 EUR/m2
  temperature:  147.5 h outside the band
          co2:  117.5 h outside the band
     humidity:  506.2 h outside the band
         lamp:    0.0 h outside the band
```

The setup (weather, the [quickstart](quickstart.md) steps) is the same as for the incumbent. Under the hood the script calls `run_season_controller(env, controller, seed)`, the same harness every reference controller runs through. It times `act`, so the decision time is measured the same way for everyone.

The thermostat's net value is higher than the incumbent's (+3.088 for the same season), but it spends many more hours outside the band. That is why the feasibility gate runs before any ranking: see [KPIs and the gate](kpis.md).

The gate judges your controller by crop rules, not by the band above: a 6-day mean temperature that follows daily light, a 24-hour heat cap, a dose of extreme temperatures, absolute limits, humid hours, and the longest humid run. Your controller may plan against these rules directly. The numbers and their sources are in [KPIs and the gate](kpis.md#crop-rules).

## Run it through the benchmark

The same class runs the full grid, scored and gated like the reference controllers, without any change to Phytotron. Name it with `--controller name=module:Class`; the module is imported from the current directory:

```
phytotron run --run-dir runs/mine --controller thermostat=examples.my_controller:Thermostat \
    --arms incumbent thermostat --years 2025 --seeds 0 --workers 2
```

The summary then lists `thermostat` next to the incumbent, with its gate verdict, net value, safety profile and place on the Pareto front. Its net value for a season equals what `examples/my_controller.py` prints for the same season: it is the same code path.

- A class needs a no-argument constructor. A controller that needs the simulator, the configuration or a perturbed model is registered as a factory instead: `--controller mine=my_module:make_controller`, where `make_controller(env, cfg, mismatch, seed)` returns the controller.
- `--model-based name` marks a controller that plans with a model. It then runs the whole mismatch ladder, and its verdict pools the seasons above mismatch 0, like the reference MPC. Without the flag it runs at mismatch 0 only, like the incumbent.
- `--seed-sensitive name` marks a controller whose season depends on the seed (a stochastic policy). Every seed then runs; by default one seed runs per weather year and mismatch level, as for the incumbent.
- The run record stores the module's file and its hash. A resume with changed controller code is refused, so a run never mixes two versions of your controller.
- Leaving out `--arms` runs the reference arms plus every registered controller.

The incumbent is always the reference and is never replaced.

## Model-based controllers: the nominal model rule

If your controller uses a model of the greenhouse, it must not plan with the simulator's own parameters. Take a perturbed copy instead:

```python
import numpy as np
from phytotron.controllers.base import NominalParameters
from phytotron.mismatch import perturb_params

true = {str(i): float(v) for i, v in enumerate(np.array(env.p).flatten())}
nominal = NominalParameters(perturb_params(true, magnitude=0.10, seed=seed))
```

Subclass `phytotron.controllers.base.PhysicsController` to have this enforced: it raises `TypeError` when given the simulator's true parameters. Report results per mismatch level, and label mismatch 0 as the reference it is (see [the glossary](README.md#glossary)).

## Run the full benchmark grid

`phytotron run` runs every arm, yours included, over every held-out season, mismatch level and seed, in parallel, and writes a gated summary (see [Run it through the benchmark](#run-it-through-the-benchmark) above). For a run inside your own script, `phytotron.grid.build_tasks` lists the seasons of the grid, `phytotron.season.run_task` runs one and returns its record, and `phytotron.summary.summarize` turns the records into the same gated summary.

## Checklist before you report a result

- `act` returns six finite values in [0, 1] (or in [-1, 1] for `"incremental"`).
- The same seed gives the same season. For MPC-style controllers, pin the math libraries to one thread (`phytotron.reproducibility.pin_threads()`, called before numpy loads); see [the MPC page](mpc.md#linear-solver-and-reproducibility).
- A model-based controller gets a nominal model, never the simulator's parameters.
- Results are labelled simulated, with the gate outcome next to the net value.
