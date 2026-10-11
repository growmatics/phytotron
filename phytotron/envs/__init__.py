"""Adapter for the GreenLight-Gym2 TomatoEnv.

The pure info-parsing helpers work without the simulator and are unit-tested;
the env factory and season runner require ``gl_gym`` (imported lazily).
"""

from phytotron.envs.glg2 import (
    VIOLATION_FIELDS,
    SeasonResult,
    make_tomato_env,
    season_result_from_infos,
)

__all__ = [
    "VIOLATION_FIELDS",
    "SeasonResult",
    "make_tomato_env",
    "season_result_from_infos",
]
