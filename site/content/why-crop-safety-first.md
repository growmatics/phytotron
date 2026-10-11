---
path: /phytotron/why-crop-safety-first
title: "Why we gate greenhouse controllers on crop safety before profit | Phytotron by Growmatics"
description: "A greenhouse climate controller can earn more by letting the crop sit too cold or too humid. Phytotron checks every simulated season against crop-safety rules from published sources before comparing money. What the first run showed."
schema: Article
schema_fields:
  headline: Why we gate greenhouse controllers on crop safety before profit
  datePublished: "2026-10-12"
  publisher: {name: Growmatics B.V., url: https://growmatics.ai}
eyebrow: Article · 12 October 2026
cite_as: "Growmatics (2026). Why we gate greenhouse controllers on crop safety before profit. https://growmatics.ai/phytotron/why-crop-safety-first"
---

# Why we gate greenhouse controllers on crop safety before profit

All results in this article are simulated.

When we started comparing greenhouse climate controllers, we did what most papers do: run each controller through the same simulated seasons and rank them on money. Revenue from the fruit, minus heating, CO2 and electricity. The ranking was clean, and it was wrong.

## The controller that wins by neglect

The simulator does price part of the harm: a cold greenhouse grows less fruit, and the revenue falls. But it prices only what its crop model contains and only inside the season it simulates. A 60-day spring season ends before slow development costs the grower the summer harvest. The crop model has no grey mould, so a week of humid nights costs nothing. It has no fruit quality, no physiological disorders, no pollination failure at the extremes. A controller that saves heat or runs no lamps can therefore come out ahead on net value while doing things no grower would accept. Our own thermostat example, twenty lines of code, does exactly that: it beats the rule-based incumbent on net value for a season while spending hundreds of hours above the humidity ceiling and never dosing CO2 or switching on a lamp.

So a ranking on money alone rewards neglect, not because the controller is clever but because the simulator cannot see the damage. Everyone in greenhouse control knows this, and every paper handles it the same way: count the hours outside a band, or add a penalty per degree outside it. That tells you a controller violated the band. It does not tell you whether the crop would have cared.

## What the literature says, and what it does not

We went looking for the number that would settle it: how much climate violation can a tomato crop take? There is no such number. What the literature has is better than a single number. It says what kind of harm follows what kind of climate.

Tomato responds to the mean temperature over a day and over several days, not to short swings. Körner and Challa built a whole control concept on that in 2003, with a six-day window and a band of four degrees around a target. Heat hurts fruit set when the daily mean stays high: Adams and colleagues put the line at 26 C in 2001. Short extremes matter as a dose: Körner and Challa weight time below 14 C or above 30 C exponentially toward the limits of 10 and 34 C. Disease follows humid hours: grey mould risk accumulates with hours above 90% humidity, and a dry break does not reset it. One long wet spell is its own risk.

Six rules, each with a source, each standing for a different harm. A season fails a class if any rule of that class breaks. A controller passes if, with 95% confidence, at most 5% of its seasons fail. Only controllers that pass are compared on money. [The rules and their sources are on one page.](/phytotron/crop-rules)

## What the first run showed

We ran our two reference controllers, a rule-based incumbent and an economic model predictive controller, on 108 simulated seasons. [The full results are here.](/phytotron/results)

Under a plain band rule, every season of both controllers failed. That is the state of the art: the band fails everyone, so it separates no one.

Under the crop rules, humidity separated them. The incumbent broke no humidity rule in any season. The MPC broke the humid-hours rule in 2 of its 100 verdict seasons, both at the largest model error, which put its upper bound just above the 5% line. The incumbent spends 488 hours per season above 85% humidity, the band's ceiling, and still breaks no crop rule: its humid periods stay below the 90% and 93% thresholds the rules use. The band could never have shown that.

Temperature failed both, for real reasons. The MPC plans against the band, not against the crop, and with model error it runs the greenhouse cold: its six-day mean fell more than 4 C below target in 77 of 104 seasons. The incumbent failed on one or two isolated 15-minute dips below 10 C per season, each when the thermal screen opened with heating at full power. A grower opens the screen in steps.

## The rule we changed, and how

That last finding is also where the method earned its keep. A single 15-minute dip failing a whole season is strict, and the source of the 10 C limit used it as a setpoint, not as a verdict. We changed the rule: protocol 2 requires 30 minutes beyond a limit, the same 30 minutes the dose rule already treats as a full dose.

We did not change it quietly. The run was scored under protocol 1 and published that way. Then it was rescored under protocol 2 from the stored climate traces, and both are reported side by side. The change moved one season out of 108 and no verdict. Every result names the protocol it was scored under, and the next run will use protocol 2.

That is the whole point of a benchmark: not that the rules are perfect, but that they are written down with their sources, applied before the results are seen, and changed only in the open.

## Bring your controller

The code is public, with the first run's season traces published alongside it. Any controller that maps the greenhouse state to heating, CO2, screens, ventilation and lamps can be scored. [How to bring yours.](/phytotron/bring-your-controller)
