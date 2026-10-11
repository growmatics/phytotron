---
path: /phytotron/results
title: "Simulated results: reference greenhouse climate controllers under crop-safety rules | Phytotron by Growmatics"
description: "Simulated benchmark results for a rule-based greenhouse climate controller and an economic MPC on 108 tomato seasons: crop-safety verdicts, net economic value by model error, where the money goes, and what the crop rules found."
schema: Dataset
schema_fields:
  name: Phytotron reference results, run of 10 October 2026
  datePublished: "2026-10-10"
  publisher: {name: Growmatics B.V., url: https://growmatics.ai}
  license: https://spdx.org/licenses/AGPL-3.0-or-later.html
  distribution: "results/2026-10-10-run-2/summary.json in the repository, at the commit named in results.json"
  isBasedOn: https://growmatics.ai/phytotron/crop-rules
eyebrow: Simulated results · reference controllers · run of 10 October 2026
publish_when: the Phytotron code is public (version 0.1); not before
cite_as: "Growmatics (2026). Phytotron reference results, run of 10 October 2026. https://growmatics.ai/phytotron/results"
---

# Simulated results: two reference controllers under the crop-safety rules

All numbers on this page are simulated. They come from one run of the Phytotron benchmark on its reference scenario, scored under the [crop-safety rules](/phytotron/crop-rules) fixed on 10 October 2026 (protocol 1), and rescored under protocol 2 at the end of the page. The run's summary files are in the repository, so anyone can check every number in the tables. The 108 per-season files with the full climate traces are published with the code, as an asset of its release ([the season trace files](https://github.com/growmatics/phytotron/releases/download/v0.2.0/phytotron-run-2026-10-10-seasons.zip)); the few numbers on this page that come from them say so.

{{run_setup}}

## Verdicts

{{verdict_table}}

No controller passes the gate, so the ranking holds only the incumbent, as the reference. The incumbent's net value is higher than the MPC's at every level of model error. The incumbent is the reference, not an arm the gate judges: 4 seasons cannot show a failure rate below 5%.

## The MPC by model error

{{mismatch_table}}

The MPC's net value falls steadily as its model gets worse, and its solver fails more often. The 24-hour mean rule and the longest-humid-run rule were never broken by any controller.

## Where the money goes

Mean per season, EUR/m2.

{{money_table}}

The incumbent spends four times as much on electricity (lamps) and grows more than twice the fruit. The MPC saves on lamps and CO2 but loses more revenue than it saves. Both lose money over these spring seasons at the configured prices; net value is revenue minus variable costs only, so the comparison between controllers is what matters, not the sign. The year matters more than the controller.

## What the crop rules found

**Humidity separates the controllers.** The incumbent broke no humidity rule. The MPC broke the humid-hours rule in 2 of its 100 verdict seasons, both at the largest model error, which puts its upper bound just above the 5% line. Under a plain band rule (more than 2 points above 85% RH for more than an hour) every season of both controllers would fail; the crop rules make the difference visible.

**Temperature fails everyone, for real reasons.**

- The MPC runs the greenhouse cold. Its 6-day mean fell more than 4 C below the light-following target in 77 of 104 seasons, and it dipped below 10 C in 95 seasons, 46 of them after the first day (from the season traces). It plans against the simulator's band (15 to 34 C), not against the crop rules, and with model error it undershoots even that.
- The incumbent dipped below 10 C in all 4 seasons: one or two isolated 15-minute steps per season, between day 1 and day 32 (from the season traces). Our reading of the traces: each time the thermal screen opened abruptly while heating was already at full power, and the cold air above the screen dropped in. Growers avoid this by opening the screen in steps.

Under protocol 1, a single 15-minute step below 10 C failed a whole season under the absolute-limit rule. That is strict: the source of the 10 C limit used it as a control setpoint, not as a verdict. Protocol 2, decided after this run was scored, requires 30 minutes or more in a row beyond a limit, the same 30 minutes the dose rule uses as a full dose.

## The same run under protocol 2

The run was rescored under protocol 2 from its stored climate traces, nothing rerun. Rule breaks and class failures per controller, before and after:

{{protocol_table}}

No verdict changes: the incumbent has 4 seasons and cannot pass a 5% gate, and the MPC breaks the dose rule in every season. The one season that moves is the incumbent's 2024 season, which had broken only the absolute-limit rule.

## Safety profile against the simulator band

Mean per season against the simulator band (15 to 34 C, 50 to 85% RH, 300 to 1600 ppm), which the reference MPC plans against and which is not gated.

{{safety_table}}

The incumbent spends 488 hours per season above 85% RH and still breaks no humidity crop rule: its humid periods stay below the 90% and 93% thresholds the rules use. This is why the band alone cannot gate.

## What this run is not

- Not a calibrated result: the scenario is a simulation, not a model of any named greenhouse.
- Not a verdict on MPC as a method: this MPC plans against the band, with a 1-hour horizon and no knowledge of the crop rules. An MPC that plans against the rules is a later arm.
- Not powered for the incumbent: 4 seasons cannot pass a 5% gate.

## Reproduce it

The run directory in the repository holds the summary, the run record (commit, package versions, input-file hashes, thread settings) and the grid; the per-season climate traces are the release asset linked above. The command that produced the run and the command that rescores the traces under new rules are in the repository's results documentation.
