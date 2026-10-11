# Results: reference controllers, run of 10 October 2026

All numbers on this page are simulated. They come from one run of `phytotron run` on the reference scenario, under the crop-safety rules fixed on 10 October 2026, protocol 1 ([KPIs and the gate](kpis.md)); the last section rescores the same run under protocol 2. The run's `summary.json`, `run_record.json` and `grid.json` are kept in `results/2026-10-10-run-2/` in this repository; the 108 per-season files with the full climate traces are held by Growmatics and will be published separately; until then, rescoring under future rules is done by Growmatics on request. The numbers below that come from those traces, not from the summary, are marked as such.

## Setup

- Phytotron commit `1fb0045`, GreenLight-Gym2 v0.2.1 (commit `8bb5b28`), CasADi 3.6.7, NumPy 1.26.4, Python 3.12, math libraries pinned to one thread.
- Scenario: simulated Venlo tomato greenhouse, Amsterdam weather, 60-day seasons from day 59, held-out years 2022 to 2025. Band and prices from `configs/benchmark.yaml`.
- Grid: the incumbent once per year (4 seasons); the economic MPC at mismatch 0 once per year (4 seasons, the labelled reference) and at 2.5, 5, 10, 15 and 20 % mismatch with 5 seeds per year (100 verdict seasons). 108 seasons, 0 failed, 136 minutes on 18 workers (from the season files).
- Checks: every season ran all 5,760 control steps (5,761 recorded states) and every action stayed in range. All 108 net values are identical to Phytotron's earlier run of the same seasons under the previous rules, which itself matched Growmatics' reference run of 24 July 2026 ("the study") season by season ([validation](validation.md), section 1).

## Verdicts

| Controller | Verdict seasons | Net value (EUR/m2), mean ± sd | Humidity: failing seasons, upper bound | Temperature: failing seasons, upper bound | Gate |
|---|---|---|---|---|---|
| Incumbent (rule-based) | 4 | −2.74 ± 5.41 | 0 of 4, 52.7 % | 4 of 4, 100 % | fail (underpowered: 4 seasons cannot show a rate below 5 %) |
| Economic MPC, mismatch above 0 | 100 | −4.80 ± 2.67 | 2 of 100, 6.2 % | 100 of 100, 100 % | fail |

No controller passes the gate, so the ranking contains only the incumbent, as the reference. The incumbent's net value is higher than the MPC's at every mismatch level.

## The MPC by mismatch level

| Mismatch | Seasons | Net value (EUR/m2) | Fruit growth (kg/m2) | Solve success | Seasons breaking: 6-day mean / dose / peak / humid hours |
|---|---|---|---|---|---|
| 0 (reference) | 4 | −3.06 ± 0.90 | 5.68 | 0.999 | 4 / 4 / 4 / 0 |
| 2.5 % | 20 | −3.43 ± 1.37 | 5.66 | 0.999 | 20 / 20 / 20 / 0 |
| 5 % | 20 | −3.44 ± 1.91 | 5.58 | 0.999 | 17 / 20 / 19 / 0 |
| 10 % | 20 | −4.01 ± 2.46 | 5.15 | 0.998 | 12 / 20 / 18 / 0 |
| 15 % | 20 | −5.63 ± 1.95 | 4.09 | 0.991 | 12 / 20 / 17 / 0 |
| 20 % | 20 | −7.52 ± 2.81 | 4.10 | 0.954 | 12 / 20 / 17 / 2 |

The MPC's net value falls steadily as its model gets worse, from −3.1 at mismatch 0 to −7.5 at 20 %, and its solver fails more often. The 24-hour mean rule and the longest-humid-run rule were never broken by any controller.

## Where the money goes

Mean per season, EUR/m2.

| Controller | Revenue | Heating | CO2 | Electricity | Net | Fruit growth (kg/m2) | Decision time (ms/step) |
|---|---|---|---|---|---|---|---|
| Incumbent | 17.69 | 8.46 | 2.00 | 9.98 | −2.74 | 11.06 | 0.2 |
| MPC (all 104 seasons) | 7.91 | 8.91 | 1.35 | 2.39 | −4.74 | 4.95 | 193.4 |

The incumbent spends four times as much on electricity (lamps) and grows more than twice the fruit. The MPC saves on lamps and CO2 but loses more revenue than it saves. Both lose money over these spring seasons at the configured prices; net value is revenue minus variable costs only, so the comparison between controllers is what matters, not the sign.

By year, incumbent: 2022 +2.04, 2023 −9.44, 2024 −6.65, 2025 +3.09. MPC: 2022 −3.84, 2023 −6.24, 2024 −5.70, 2025 −3.16. The year matters more than the controller.

## What the crop rules found

**Humidity separates the controllers.** The incumbent broke no humidity rule. The MPC broke the humid-hours rule in 2 of 100 verdict seasons, both at 20 % mismatch, which puts its upper bound at 6.2 %, just above the 5 % line. Under the older banded rule (more than 2 points above 85 % RH for more than an hour) every season of both controllers failed; the crop rules make the difference visible. Mean night-time humidity deficit: incumbent 2.39 g/m3, MPC 2.32 g/m3.

**Temperature fails everyone, for real reasons.**

- The MPC runs the greenhouse cold. Its 6-day mean fell more than 4 C below the light-following target in 77 of 104 seasons. It dipped below 10 C in 95 seasons, and in 46 of them after the first day (from the traces), so this is not only a cold start. Its dose of extremes exceeded the limit in every season. It plans against the simulator band (15 to 34 C), not against the crop rules, and with model error it undershoots even that.
- The incumbent dipped below 10 C in all 4 seasons: one or two isolated 15-minute steps per season, between day 1 and day 32 (from the traces). Our reading of the traces: each time the thermal screen opened abruptly while heating was already at full power, and the cold air above the screen dropped in. Growers avoid this by opening the screen in steps. Its dose rule broke in 3 seasons; its 6-day mean never left the band.
- Two MPC seasons went above 34 C (from the traces).

Under protocol 1, a single 15-minute step below 10 C failed a whole season under the peak rule. That is strict: the source of the 10 C limit (Körner and Challa 2003) used it as a control setpoint, not as a verdict. Protocol 2 (decided 10 October 2026, after this run was scored; see the protocol history in [kpis.md](kpis.md)) requires 30 minutes or more in a row beyond a limit, the same 30 minutes the dose rule uses as a full dose.

## The same run under protocol 2

Rescored from the stored climate traces with `phytotron run --run-dir runs/full-2 --rescore`, which writes `summary-rescored-<commit>.json`; that file is kept here as `results/2026-10-10-run-2/summary-protocol-2.json`. Nothing was rerun.

| Controller | Seasons | Peak breaks, protocol 1 → 2 | Dose breaks | Temperature class failures, protocol 1 → 2 | Humidity class failures |
|---|---|---|---|---|---|
| Incumbent | 4 | 4 → 0 | 3 | 4 → 3 | 0 |
| MPC | 104 | 95 → 93 | 104 | 104 → 104 | 2 |

No verdict changes: the incumbent has 4 seasons and cannot pass a 5 % gate (its temperature upper bound moves from 100 % to 98.7 %), and the MPC breaks the dose rule in every season. The one season that moves is the incumbent's 2024, which had broken only the peak rule.

## Safety profile

Mean per season against the simulator band (15 to 34 C, 50 to 85 % RH, 300 to 1600 ppm).

| Controller | Hours outside the temperature band | Temperature degree-hours (C h) | Worst temperature excursion (C) | Hours above 85 % RH | RH degree-hours (% h) | Worst RH excursion (points) | CO2 seasons with any excursion |
|---|---|---|---|---|---|---|---|
| Incumbent | 60 | 73 | 6.0 | 488 | 852 | 10.9 | 0 of 4 |
| MPC | 448 | 480 | 13.4 | 387 | 900 | 12.0 | 4 of 104 |

The incumbent spends 488 hours per season above 85 % RH and still breaks no humidity crop rule: its humid periods stay below the 90 % and 93 % thresholds the rules use. This is why the band alone cannot gate.

## What this run is not

- Not a calibrated result: the scenario is a simulation, not a model of any named greenhouse.
- Not a verdict on MPC as a method: this MPC plans against the band, with a 1-hour horizon and no knowledge of the crop rules. An MPC that plans against the rules is a later arm.
- Not powered for the incumbent: 4 seasons cannot pass a 5 % gate. The incumbent is the reference, not an arm the gate judges.

## Files

| | |
|---|---|
| Run script | `phytotron run --run-dir runs/full-2 --workers 18` |
| Summary, run record and grid | `results/2026-10-10-run-2/` (copied from `runs/full-2/`) |
| Rules | `configs/benchmark.yaml`, `gate.classes` |
| Rescoring under new rules | `phytotron run --run-dir runs/full-2 --rescore` |
