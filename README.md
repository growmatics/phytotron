# Phytotron by Growmatics

**An open, simulated benchmark for greenhouse climate control.**

Project page, method and roadmap: **[growmatics.ai/phytotron](https://growmatics.ai/phytotron)**

> Status: the code arrives here as version 0.1 in Q4 2026. Until then this page says what Phytotron is and why it exists. Watch the repository to be notified.

## Why

Papers on greenhouse climate control, from rule-based setpoints to model predictive control and reinforcement learning, each use their own greenhouse model, weather, prices and constraints. Their numbers cannot be put side by side. And in our reading of the literature, none of them says how much climate violation a crop can take: violations are counted or penalised, never judged against a crop-safety standard.

A controller can earn more by letting the crop sit too cold, too hot or too humid. Without a standard, that looks like a win.

## What

Phytotron is a test bench. Every controller runs through the same simulated tomato seasons and is scored the same way:

1. **Same seasons for everyone.** A simulated Venlo tomato greenhouse, built on the GreenLight model through GreenLight-Gym2, with Amsterdam weather, on held-out weather years the controller was never tuned on.
2. **Model error on purpose.** A controller that plans with a model gets one with deliberately wrong parameters, at several levels. Planning with the simulator's own model is reported only as a labelled reference.
3. **Crop safety first.** Each season is checked against six pass/fail rules on temperature and humidity, each taken from a published horticultural source. A controller passes if, with 95% confidence, at most 5% of its seasons break a rule. [The rules and their sources.](https://growmatics.ai/phytotron/crop-rules)
4. **Then economics.** Controllers that pass are compared on net economic value (fruit revenue minus heating, CO2 and electricity, in EUR per m2) and on decision time, against a rule-based reference controller.

All Phytotron results are simulated. Phytotron is a benchmark and simulation tool, not a controller product and not a model of any named greenhouse.

## Why open

A yardstick is only useful if anyone can rerun it. The scenario, the scoring, the crop-safety rules with their sources, and the reference controllers will all be in this repository, under AGPL-3.0-or-later, the same license as the simulator it builds on. Claims like "our controller saves 10% energy" become something anyone can check.

## Take part

Rule-based, MPC or RL: anything that maps the greenhouse state to heating, CO2, screens, ventilation and lamps can be scored. If your group works on greenhouse control and wants to be among the first scored when 0.1 is out, open an issue or start a discussion here.

## Built on

The [GreenLight](https://github.com/davkat1/GreenLight) greenhouse and tomato model (Katzin et al., Wageningen University & Research) through [GreenLight-Gym2](https://github.com/BartvLaatum/GreenLight-Gym2) (van Laatum et al.). We thank both teams for publishing their work openly.

## Who

[Growmatics B.V.](https://growmatics.ai), Netherlands, develops and sponsors Phytotron and keeps it free.
