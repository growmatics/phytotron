# Source of these files

Verbatim copies from GreenLight-Gym2 release v0.2.1, commit
`8bb5b282ffa4e3feb2fb9b9e41e1027d01a2cd24` (2025-08-18),
https://github.com/BartvLaatum/GreenLight-Gym2, by Bart van Laatum and co-authors,
licensed AGPL-3.0. Phytotron is AGPL-3.0-or-later, so the copies keep their licence.

| File here | Upstream path | SHA-256 |
|---|---|---|
| `envs/TomatoEnv.yml` | `gl_gym/configs/envs/TomatoEnv.yml` | `974a734e020be3a068b35ff3afd9b995d544309d70583577ffb3467b895f4c0d` |
| `agents/rule_based.yml` | `gl_gym/configs/agents/rule_based.yml` | `88546ddbe25cc2033eca95de08f4d957ef2d0ea8f54195242496437584c02463` |

Why they are here: the pip wheel of GreenLight-Gym2 ships no configuration files, and
the benchmark reads these two (the scenario, and the incumbent's parameters).

Do not edit them. The benchmark's band and prices are set in `configs/benchmark.yaml`
and passed to the simulator at run time; the test suite checks that the copies still
match the hashes above and that the two sources agree.
