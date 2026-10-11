---
path: /phytotron/bring-your-controller
title: "Bring your greenhouse climate controller to the Phytotron benchmark | Phytotron by Growmatics"
description: "How to run your own greenhouse climate controller, rule-based, MPC or reinforcement learning, through the Phytotron benchmark: a two-method interface, six controls, one command for the full simulated grid, and the rule for model-based controllers."
schema: TechArticle
schema_fields:
  headline: Bring your greenhouse climate controller to the Phytotron benchmark
  datePublished: "2026-10-11"
  publisher: {name: Growmatics B.V., url: https://growmatics.ai}
  proficiencyLevel: Beginner
eyebrow: Guide · for controller authors
publish_when: the Phytotron code is public (version 0.1); not before
---

# Bring your controller

Any controller that maps the greenhouse state to the six controls can be scored by Phytotron like the reference controllers, over the same simulated seasons, under the same [crop-safety rules](/phytotron/crop-rules), without changing anything in Phytotron. This page is the short version of the guide in the repository.

## Two methods

Your controller is a Python class with two methods. `reset(seed)` is called once at the start of each season: clear any internal state and seed any randomness there, so the same season gives the same result. `act(ctx)` is called every control step (15 minutes) and returns the six controls.

`ctx` carries what a controller might need: the observation vector a learned policy sees, the simulator's full state for model-based controllers, the weather forecast for the next 24 steps, the step index, and the live environment. Read what you need and ignore the rest.

## Six controls

Heating, CO2 dosing, thermal screen, ventilation, lamps, blackout screen, each a value from 0 to 1. A controller returns either the controls themselves (rule-based controllers and MPC do this) or changes to them, which is what reinforcement-learning policies trained on GreenLight-Gym2 usually return; one attribute on your class says which.

## One command for the whole grid

Point `phytotron run` at your class by name. It runs your controller over every held-out season next to the rule-based incumbent, checks every season against the crop-safety rules, and writes the same summary the reference controllers get: gate verdict with the failing count and bound, net value, safety profile, and the place on the Pareto front of net value against decision time.

- A controller that plans with a model is marked as model-based. It then runs the whole model-error ladder, and its verdict pools the seasons above zero error, like the reference MPC.
- A stochastic controller is marked as seed-sensitive, so every seed runs.
- The run record stores your module's file and its hash. A resumed run with changed controller code is refused, so a result never mixes two versions of your controller.
- A trained reinforcement-learning policy runs through a wrapper; Phytotron ships no trained policy.

## The rule for model-based controllers

A controller that uses a model of the greenhouse must not plan with the simulator's own parameters. Phytotron hands it a perturbed copy at several error levels, because a real greenhouse never matches its model. Planning with the simulator's own parameters is the "inverse crime" and is reported only as a labelled reference. A base class enforces this for you.

## Before you report a result

- Every number is simulated, and says so.
- The gate outcome stands next to the net value, never apart from it.
- The same seed gives the same season. Controllers that solve optimisation problems need the math libraries pinned to one thread; the repository explains how.
- Results are reported per model-error level, with zero error labelled as the reference it is.

## Take part

Open an issue or start a discussion on [github.com/growmatics/phytotron](https://github.com/growmatics/phytotron) if you want your controller among the first scored. The full guide, the 20-line template and the exact commands are in the repository's documentation.
