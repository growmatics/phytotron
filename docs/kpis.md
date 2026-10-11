# KPIs and the feasibility gate

How Phytotron scores a controller. Every number here is computed on simulated seasons from the GreenLight-Gym2 simulator, not on a real greenhouse. Constants come from `configs/benchmark.yaml`.

## The order: gate first, then rank

A controller is scored in two steps.

1. **Feasibility gate.** Does it keep the crop climate inside the allowed band often enough? A controller that fails is reported, but not ranked.
2. **Ranking.** Among controllers that pass, compare net economic value against decision time.

The order matters because the two goals conflict. A controller can raise its margin by letting the climate drift: less heating, less ventilation, more humidity. If violations were only a penalty inside one score, a large enough margin would buy any amount of crop risk, and the result would depend on an arbitrary penalty weight. A grower does not trade disease risk for euros at a fixed rate, so Phytotron does not either.

An example. Over the 2025 season (start day 59, 60 days, Amsterdam weather) the naive thermostat in [`examples/my_controller.py`](../examples/my_controller.py) and the incumbent give:

| | Net economic value (EUR/m2) | Hours outside the band: temperature / CO2 / humidity |
|---|---|---|
| Thermostat | +4.809 | 147.5 / 117.5 / 506.2 |
| Incumbent | +3.088 | 43.0 / 0.0 / 448.8 |

On net value alone the thermostat wins. It does so by running a worse climate.

## A season

The unit of evaluation is one growing season: a Venlo-type tomato greenhouse, Amsterdam weather of one real year, starting on day 59 (about 1 March) and lasting 60 days, with a control step of 900 s (5,760 steps). Held-out seasons are 2022 to 2025. Training years (2014 to 2021) exist for controllers that learn.

## Net economic value

Fruit revenue minus variable operating cost over the season, in EUR per m2 of greenhouse floor.

- **Revenue**: fruit dry-matter growth times the fruit price (1.6 EUR/kg fresh weight, at a dry-matter fraction of 0.065).
- **Variable cost**: heating energy (0.09 EUR/kWh), CO2 dosing (0.30 EUR/kg), and lamp electricity (0.30 EUR/kWh).
- **Not included**: fixed costs (greenhouse, installations). They do not depend on the controller, so they do not change any comparison.
- **Not subtracted**: climate violations. They go to the gate instead (see above).

The simulator computes revenue and each cost per step (`revenue`, `heat_cost`, `co2_cost` and `elec_cost` in GreenLight-Gym2's step output) and Phytotron sums them over the season. Every result shows the four parts, so a reader sees where a gain comes from: the incumbent's 2025 season is revenue 17.687 minus heating 7.623, CO2 2.070 and electricity 4.906, giving +3.088 EUR/m2.

**Fruit growth** is revenue divided by the fruit price: the growth of fruit fresh weight on the plants over the season, in kg/m2 (11.05 for the incumbent in 2025). It is the "equal yield" side of a claim like "less gas at equal yield". It counts fruit grown, not fruit harvested. Net value can be negative: a 60-day season from early March is heating-heavy. The incumbent's 2023 season, for example, gives -9.44 EUR/m2 (see the [validation page](validation.md)).

## The allowed climate band

Four climate classes are tracked at every step. Each step records the distance outside the band, zero when inside.

| Class | Band | Unit of the distance |
|---|---|---|
| Air temperature | 15 to 34 C | C |
| CO2 | 300 to 1600 ppm | ppm |
| Relative humidity | 50 to 85 % | percentage points |
| Lamp use | lamps off after 20:00 | 1 per step with lamps on |

The 85 % humidity ceiling is a stand-in for disease risk (fungal diseases such as Botrytis need long periods of high humidity).

The band and the prices have one source, `configs/benchmark.yaml`. Phytotron passes them to the simulator when it builds a season (`phytotron.envs.glg2.scenario_overrides`), so the distances and costs the simulator computes and the targets the MPC plans for are the same numbers.

## Safety profile

One violation rate hides how bad the violations are, so each class gets a profile over the seasons:

- **Incidence**: the fraction of seasons with any step outside the band.
- **Mean duration**: hours outside the band per season.
- **Degree-hours**: distance times duration, summed per season and averaged (C.h, ppm.h, %RH.h). A half-degree overshoot for ten hours and a five-degree overshoot for one hour both give 5 C.h.
- **Worst excursion**: the largest single-step distance in any season.

## The feasibility gate

The gate asks: with 95 % confidence, does this controller violate the band in at most 5 % of seasons?

- **Trial unit: the season.** Steps within a season are strongly correlated, so counting steps would overstate the evidence.
- **The test.** With `k` violating seasons out of `n`, compute the one-sided 95 % Clopper-Pearson upper bound on the violation rate. The controller passes if that bound is at most 5 % (`epsilon = 0.05`, `delta = 0.05`).
- **What it takes.** At 60 seasons, passing needs zero violating seasons (upper bound 4.9 %); one violating season gives 7.7 % and fails. One violating season is tolerated only from 93 seasons on. With fewer than 59 seasons even a clean record cannot pass, so the minimum is set at 60 (`n_seasons_min`) and a smaller sample is flagged **underpowered**.
- **Crashes count.** A season the controller cannot finish counts as a violating season: a failure while the controller is built, during the season, or in the checks after it. A controller cannot improve its record by failing on its hardest seasons. A season the harness could not start (configuration, simulator, reset) never reached the controller: the summary lists it as not started, it counts in no gate, and the arm is judged on the seasons that ran. This split applies from version 0.1.0; the run of 10 October 2026 had no failure of either kind.
- **Gated classes.** Humidity and temperature are gated. CO2 and lamp use are reported in the safety profile, never gated. The list, with each class's kind and rules, is `gate.classes` in the config; the gate reads it.
- **Which seasons.** For a model-based controller (the MPC) the verdict pools every season with mismatch above 0. Mismatch 0 seasons, where the controller plans with the simulator's own parameters, are reported as a labelled reference but never count towards the verdict (see [Model mismatch](#model-mismatch)). For a controller without a model, the verdict covers all its seasons. Every mismatch level is also gated on its own and reported, flagged underpowered when it has fewer than 60 seasons.

Each season gets a yes or no per gated class. Three rules give that answer, and all three are reported:

- **Crop rules** (the primary gate, which decides the ranking). Measures taken from the crop literature, read from the season's air temperature and humidity at every step. A season breaks a class if it breaks any of the class's rules.
- **Strict**: a season violates a class if it is outside the band at any single step.
- **Banded**: a season violates a class only if it stays more than 2 units outside the band (percentage points for humidity, C for temperature) for more than one hour in a row. Kept as a sensitivity check.

### Crop rules

Tomato does not respond to climate the way a band suggests. Its growth and fruit set follow the 24-hour and multi-day mean temperature, plus a dose of extreme temperatures; short swings within limits do no harm (de Koning 1988; Körner and Challa 2003; Peet et al. 1998). Disease risk follows the number of humid hours, and dry breaks do not reset it (Eden et al. 1996; Yunis et al. 1994). The rules follow these sources, plus an absolute peak limit in the style of the UK overheating standard CIBSE TM52, which also combines a cumulative and a peak criterion.

| Class | Rule | A season breaks it when | Source |
|---|---|---|---|
| Temperature | `mean_6d` | the 6-day mean is more than 4 C away from the 6-day mean target. Each day's target is 17.5 C plus 1.5 C per 1000 J/cm2 of that day's outdoor radiation | Körner and Challa 2003 (6 days, ±4 C); Het Nieuwe Telen Tomaat 2010 (target) |
| Temperature | `mean_24h` | the 24 h rolling mean exceeds 26 C | Adams et al. 2001 |
| Temperature | `dose` | the dose of extremes in any 24 h exceeds 1. Steps below 14 C or above 30 C add to it, weighted as in Körner and Challa's Eq. 3 and 4 (exponents 160 and 30): 0 at 14 or 30 C, 1 at 10 or 34 C. 30 minutes at 10 or 34 C is a full dose | Körner and Challa 2003 |
| Temperature | `peak` | the air temperature stays below 10 C or above 34 C for 30 minutes or more in a row (two consecutive steps) | Körner and Challa 2003 (limits, and the 30 minute full-dose time); Vanthoor 2011 |
| Humidity | `humid_hours` | the 7-day rolling mean of daily hours above 90 % RH exceeds 7 h/day | Yunis et al. 1994; Sentelhas et al. 2008 |
| Humidity | `longest_run` | RH stays above 93 % for more than 48 h in a row | Körner 2003 |

One broken rule fails the season for that class: each rule stands for a different harm, and a lethal peak is never excused by good means. On the held-out springs (2022 to 2025, the weather files named by hash in the run record) the 6-day target runs from 18.3 to 20.7 C, so the band reaches from about 14.3 to 24.7 C. The rules nest: peak limits outside the dose onsets, the dose onsets outside the 6-day band, the 24 h cap above it.

**Protocol version.** Every change to a rule raises `gate.protocol_version` in the config, and every summary and page prints the version it was scored under. Results under different versions are never mixed; after a change, stored runs are rescored and both versions are reported side by side.

| Version | Date | Change | Why |
|---|---|---|---|
| 1 | 2026-10-10 | The six rules as fixed from the sources. The peak rule failed a season at any single step beyond 10 or 34 C. | First fixed protocol. |
| 2 | 2026-10-10 | The peak rule needs 30 minutes or more in a row beyond a limit (two consecutive 15-minute steps). Nothing else changed. | One 15-minute transient decided a 60-day season, and the source uses the limits as setpoints, not as a verdict. The 30 minutes is Körner and Challa's full-dose time, already used by the dose rule. Decided after the run of 10 October 2026 was scored under version 1; that run is reported under both ([results](results.md)). |

What the numbers are, and where they come from:

- **Fixed before any result was scored under them**, from a check of the primary sources, so they are not chosen to make a controller pass.
- **The sources do not give one season-wide limit**, and Phytotron does not invent one.
- **Two rules come from other crops.** No published rule exists for tomato humid hours, so `humid_hours` adapts a cucumber leaf-wetness rule. The simulator has no leaf wetness, so RH above 90 % stands in for it, the best simple proxy in Sentelhas et al. (2008). `longest_run` comes from chrysanthemum, and its 48 h is Körner's modelling choice.
- **The dose is a design choice.** In Körner and Challa it triggers a control action; Phytotron uses it as a pass/fail rule and sums it over any 24 h instead of their 6-hour reset.

Every summary also reports, per season, the mean night-time humidity deficit in g/m3 (how much more water the air could hold, at steps with no outdoor light). Dutch growers steer on it. It is reported, not gated. Rule kinds are in `phytotron/metrics/crop_rules.py`; the numbers are in `gate.classes` in the config, and a test checks that this table matches them.

### What a controller is judged on

Every controller is judged by these crop rules, whatever it plans against. A new controller may plan against them directly. The reference MPC does not: it plans against the simulator's band (15 to 34 C, RH at most 85 %), as in the benchmark it reproduces, so its numbers stay comparable with that benchmark. The band and the rules differ. A week at 15 C is inside the band but breaks `mean_6d`; holding RH at 85 % is stricter than the humidity rules ask. The reference MPC's verdict should be read with that in mind.

### Reading a verdict

Each verdict gives the number of failing seasons, the number of seasons, and the 95 % upper bound on the failing-season rate (`gate_crop_counts` in the summary), next to pass or fail. With fewer than 93 seasons a pass needs zero failing seasons; from 93 seasons one is tolerated, from 124 two, from 153 three. So a fail with one failing season out of 60 and a fail with 60 out of 60 are very different results, and the summary shows which.

### Why the band rules are not the primary gate

The strict gate saturates in practice: in Growmatics' earlier benchmark runs every season of every controller crossed 85 % RH at some step, so it passes no one. The banded rule separates the controllers' violation rates but passes no one either. In Phytotron's full run of 2026-10-10, every MPC season and every incumbent season broke the banded humidity rule. Both band rules count how far and how long the climate leaves a band, and the crop literature does not measure harm that way. They stay in every summary, so readers can see how the verdict depends on the rule.

## Decision time

The mean wall-clock time per control step that the controller spends deciding, in seconds. It is the second ranking axis, because a controller that needs a server farm is not the same offer as one that runs on a climate computer. The value depends on the machine, so it compares controllers run on the same machine; the run record stores the hardware.

## Solve success (MPC only)

The share of optimisation problems the MPC solved or recovered a usable answer from. When it fails, it holds the last control. A controller that often falls back is less trustworthy than its net value suggests. See [the MPC page](mpc.md#solver-recipe).

## Ranking: the Pareto front

Among controllers that pass the primary gate (the crop rules) on every gated class (the incumbent is always included as the reference), Phytotron reports the Pareto front over the two axes: higher net value, lower decision time. A controller is on the front if no other controller is at least as good on both axes and better on one. There is no single winner by design: a cheaper, slightly worse controller and a better, slower one can both be on the front.

The run summary also gives a hypervolume, the area of the front after scaling both axes to [0, 1]. It mixes euros and seconds on one scale, so treat it as a diagnostic, not a score.

## Model mismatch

Model-based controllers (the MPC) plan with a model. In the benchmark they never get the simulator's own parameters: they get a copy perturbed by a mismatch level from 0 to 20 % (`mismatch.sweep_magnitudes`). Mismatch 0 is reported only as a labelled reference, because a controller that plans with the exact simulator is graded on its own answer key (see "inverse crime" in the [glossary](README.md#glossary)). Results are reported per mismatch level, so the reader sees how fast a controller degrades as its model gets worse.

## The grid, and what it can show

The benchmark grid (`phytotron run`) runs each held-out year once per condition:

- **Incumbent**: one season per year, 4 in total. It has no model and the simulator is deterministic, so more seeds would repeat the same season. Its gate verdict is therefore always underpowered: 4 seasons cannot show a violation rate below 5 %.
- **MPC**: at mismatch 0, one season per year (the model equals the simulator's for every seed); at each of the 5 levels above 0, 5 seeds per year, each seed a different perturbed model. That is 100 verdict seasons and 4 reference seasons.

So the gate can reach a verdict for the MPC, but only describe the incumbent. The incumbent stays in every ranking as the reference.

## Where the code is

| KPI | Code |
|---|---|
| Net value and per-step distances | `phytotron/envs/glg2.py`, `season_result_from_infos` |
| Safety profile, banded rule | `phytotron/metrics/safety.py` |
| Crop rules | `phytotron/metrics/crop_rules.py` |
| Gate (Clopper-Pearson) | `phytotron/metrics/gate.py` |
| Pareto front, hypervolume | `phytotron/metrics/pareto.py` |
| Per-arm aggregation | `phytotron/benchmark.py`, `BenchmarkResult` |
| Grid and summary | `phytotron/runner.py` |

## Sources for the crop rules

- Adams, S.R., Cockshull, K.E. and Cave, C.R.J. (2001). Effect of temperature on the growth and development of tomato fruits. *Annals of Botany* 88:869-877. https://doi.org/10.1006/anbo.2001.1524
- de Gelder, A., Warmenhoven, M. and Grootscholten, M. (2010). *Het Nieuwe Telen Tomaat 2010.* Wageningen UR Glastuinbouw, rapport GTB-1178. https://edepot.wur.nl/247906
- Sentelhas, P.C. et al. (2008). Suitability of relative humidity as an estimator of leaf wetness duration. *Agricultural and Forest Meteorology* 148:392-400. https://doi.org/10.1016/j.agrformet.2007.09.011
- Vanthoor, B.H.E. et al. (2011). A methodology for model-based greenhouse design: Part 2, description and validation of a tomato yield model. *Biosystems Engineering* 110:378-395. https://doi.org/10.1016/j.biosystemseng.2011.08.005
- de Koning, A.N.M. (1988). More efficient use of base load heating with a temperature integrating control programme. Effect on development, growth and production of tomato. *Acta Horticulturae* 229:233-238. https://doi.org/10.17660/ActaHortic.1988.229.22
- Körner, O. and Challa, H. (2003). Design for an improved temperature integration concept in greenhouse cultivation. *Computers and Electronics in Agriculture* 39(1):39-59. https://doi.org/10.1016/S0168-1699(03)00006-1
- Körner, O. (2003). *Crop based climate regimes for energy saving in greenhouse cultivation.* PhD thesis, Wageningen University. https://edepot.wur.nl/27517
- Peet, M.M., Sato, S. and Gardner, R.G. (1998). Comparing heat stress effects on male-fertile and male-sterile tomatoes. *Plant, Cell & Environment* 21:225-231. https://doi.org/10.1046/j.1365-3040.1998.00281.x
- Eden, M.A., Hill, R.A., Beresford, R. and Stewart, A. (1996). The influence of inoculum concentration, relative humidity, and temperature on infection of greenhouse tomatoes by *Botrytis cinerea*. *Plant Pathology* 45:795-806. https://doi.org/10.1046/j.1365-3059.1996.d01-163.x
- Yunis, H., Shtienberg, D., Elad, Y. and Mahrer, Y. (1994). Qualitative approach for modelling outbreaks of grey mould epidemics in non-heated cucumber greenhouses. *Crop Protection* 13(2):99-104. https://doi.org/10.1016/0261-2194(94)90158-9
- CIBSE (2013). *TM52: The limits of thermal comfort: avoiding overheating in European buildings.* Chartered Institution of Building Services Engineers, London.
