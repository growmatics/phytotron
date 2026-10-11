"""Scenario files Phytotron ships, so a run needs no simulator checkout.

``glg2_v0_2_1/`` holds verbatim copies of the two GreenLight-Gym2 v0.2.1 configuration
files the benchmark reads: the ``TomatoEnv`` scenario (greenhouse, crop, band, prices)
and the rule-based incumbent's parameters. The pip wheel of GreenLight-Gym2 ships
neither. ``SOURCE.md`` next to them gives the origin and licence; ``UPSTREAM`` below
records their SHA-256 so a changed copy is caught (``check_scenario_files``).

The benchmark's own band and prices live in ``configs/benchmark.yaml`` and are passed to
the simulator at run time (``phytotron.envs.glg2.scenario_overrides``); the test suite
checks that the two agree.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

GLG2_RELEASE = "v0.2.1"
GLG2_COMMIT = "8bb5b282ffa4e3feb2fb9b9e41e1027d01a2cd24"
GLG2_URL = "https://github.com/BartvLaatum/GreenLight-Gym2"

_ROOT = Path(__file__).resolve().parent / "glg2_v0_2_1"

# Path inside the GreenLight-Gym2 checkout -> SHA-256 of the file at GLG2_COMMIT.
UPSTREAM: dict[str, str] = {
    "gl_gym/configs/envs/TomatoEnv.yml": (
        "974a734e020be3a068b35ff3afd9b995d544309d70583577ffb3467b895f4c0d"
    ),
    "gl_gym/configs/agents/rule_based.yml": (
        "88546ddbe25cc2033eca95de08f4d957ef2d0ea8f54195242496437584c02463"
    ),
}
_LOCAL: dict[str, Path] = {
    "gl_gym/configs/envs/TomatoEnv.yml": _ROOT / "envs" / "TomatoEnv.yml",
    "gl_gym/configs/agents/rule_based.yml": _ROOT / "agents" / "rule_based.yml",
}


def glg2_config_dir() -> Path:
    """The shipped ``envs`` directory, laid out as GreenLight-Gym2's own (``../agents``
    holds the incumbent's parameters)."""
    return _ROOT / "envs"


def scenario_files() -> dict[str, Path]:
    """Upstream path -> shipped copy."""
    return dict(_LOCAL)


def check_scenario_files() -> list[str]:
    """Shipped files whose SHA-256 differs from the recorded upstream hash (empty if
    none). A non-empty list means a copy was edited: scenario values belong in
    ``configs/benchmark.yaml``, never in these files."""
    return [
        upstream
        for upstream, path in _LOCAL.items()
        if hashlib.sha256(path.read_bytes()).hexdigest() != UPSTREAM[upstream]
    ]
