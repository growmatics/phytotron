"""The example controller in ``examples/my_controller.py`` meets the Controller protocol."""

import importlib.util
from pathlib import Path

import numpy as np

from phytotron.controllers.base import ControlContext, Controller

_PATH = Path(__file__).resolve().parent.parent / "examples" / "my_controller.py"
_spec = importlib.util.spec_from_file_location("my_controller", _PATH)
assert _spec is not None and _spec.loader is not None
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def _ctx(t_air: float) -> ControlContext:
    state = np.zeros(28)
    state[example.AIR_TEMP] = t_air
    return ControlContext(obs=np.zeros(22), state=state, forecast=np.zeros((4, 10)), timestep=0)


def test_thermostat_is_a_controller() -> None:
    assert isinstance(example.Thermostat(), Controller)


def test_thermostat_heats_when_cold_and_ventilates_when_hot() -> None:
    c = example.Thermostat(heat_c=18.0, vent_c=24.0, band_c=2.0)
    c.reset(0)
    cold, mild, hot = c.act(_ctx(15.0)), c.act(_ctx(21.0)), c.act(_ctx(25.0))
    assert cold[example.HEATING] == 1.0
    assert cold[example.VENTILATION] == 0.0
    assert not mild.any()
    assert hot[example.VENTILATION] == 0.5
    assert all(u.shape == (6,) and (0.0 <= u).all() and (u <= 1.0).all() for u in (cold, hot))
