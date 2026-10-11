"""Run a controller from the user's own code through the benchmark, by name.

A controller is named on the command line as ``name=module:attr``::

    phytotron run --run-dir runs/mine --controller thermostat=my_controller:Thermostat

``module`` is imported with the current directory on the path, so a file next to the
shell works. ``attr`` is either a controller class with a no-argument constructor, or a
factory ``(env, cfg, mismatch, seed) -> controller`` for controllers that need the
simulator, the benchmark configuration, or a perturbed model.

The spec string travels inside each task, so worker processes build the controller
themselves. The run record stores the module file and its hash, so a result names the
exact controller code and a resume refuses changed code.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SPEC = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<target>[A-Za-z_][\w.]*:[A-Za-z_]\w*)$")


@dataclass(frozen=True)
class ArmSpec:
    """A user controller registered as a benchmark arm."""

    name: str
    target: str  # "module:attr"
    model_based: bool = False  # runs the mismatch ladder, verdict pooled above mismatch 0
    seed_sensitive: bool = False  # every seed runs (default: one seed, like the incumbent)

    def seed_invariant(self, mismatch: float) -> bool:
        """True when the seed cannot change the season for this arm."""
        if self.seed_sensitive:
            return False
        return not self.model_based or mismatch == 0.0


def parse_spec(text: str) -> tuple[str, str]:
    """``"name=module:attr"`` to ``(name, "module:attr")``."""
    match = _SPEC.match(text.strip())
    if not match:
        raise ValueError(f"controller spec must look like name=module:Class, got {text!r}")
    return match["name"], match["target"]


def parse_specs(
    texts: list[str] | None,
    *,
    model_based: list[str] | None = None,
    seed_sensitive: list[str] | None = None,
    reserved: tuple[str, ...] = (),
) -> dict[str, ArmSpec]:
    """The custom arms from the command line, keyed by name, checked for clashes."""
    arms: dict[str, ArmSpec] = {}
    for text in texts or []:
        name, target = parse_spec(text)
        if name in reserved:
            raise ValueError(f"{name!r} is a built-in arm; choose another name")
        if name in arms:
            raise ValueError(f"controller {name!r} given twice")
        arms[name] = ArmSpec(
            name,
            target,
            model_based=name in (model_based or []),
            seed_sensitive=name in (seed_sensitive or []),
        )
    for flag, names in (("--model-based", model_based), ("--seed-sensitive", seed_sensitive)):
        unknown = sorted(set(names or []) - set(arms))
        if unknown:
            raise ValueError(f"{flag} names {unknown} that are not registered with --controller")
    return arms


def load_target(target: str) -> Any:
    """Import ``module:attr`` with the current directory on the path."""
    module_name, attr = target.split(":", 1)
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise ValueError(f"cannot import {module_name!r} for controller {target!r}: {err}") from err
    try:
        return getattr(module, attr)
    except AttributeError as err:
        raise ValueError(f"{module_name!r} has no attribute {attr!r}") from err


def make_factory(obj: Any) -> Callable[[Any, Any, float, int], Any]:
    """A factory ``(env, cfg, mismatch, seed) -> controller`` from a class or a function.

    A class is built with no arguments, once per season. A function is used as the
    factory itself and must take the four arguments.
    """
    if inspect.isclass(obj):
        cls = obj

        def from_class(env: Any, cfg: Any, mismatch: float, seed: int) -> Any:
            return cls()

        return from_class
    if callable(obj):
        return obj
    raise ValueError(f"controller target must be a class or a factory function, got {obj!r}")


def build_custom(target: str, env: Any, cfg: Any, *, mismatch: float, seed: int) -> Any:
    """A ready controller for one season from a ``module:attr`` target."""
    controller = make_factory(load_target(target))(env, cfg, mismatch, seed)
    for method in ("reset", "act"):
        if not callable(getattr(controller, method, None)):
            raise ValueError(f"controller {target!r} has no {method}() method")
    return controller


def target_source(target: str) -> dict[str, str]:
    """The file a target lives in and its SHA-256, for the run record."""
    module = inspect.getmodule(load_target(target))
    path = getattr(module, "__file__", None)
    if not path:
        return {"file": "unknown", "sha256": "unknown"}
    file = Path(path).resolve()
    try:
        shown = file.relative_to(Path.cwd())  # keep the user's home out of run records
    except ValueError:
        shown = file
    return {"file": str(shown), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}


def record_specs(arms: Mapping[str, ArmSpec]) -> dict[str, dict[str, Any]]:
    """The custom arms as plain data for ``run_record.json``."""
    return {
        name: {
            "target": spec.target,
            "model_based": spec.model_based,
            "seed_sensitive": spec.seed_sensitive,
            **target_source(spec.target),
        }
        for name, spec in arms.items()
    }


def specs_from_record(run_record: Mapping[str, Any]) -> dict[str, ArmSpec]:
    """The custom arms a run was launched with, from its run record."""
    return {
        name: ArmSpec(
            name, data["target"], bool(data["model_based"]), bool(data.get("seed_sensitive", False))
        )
        for name, data in (run_record.get("custom_arms") or {}).items()
    }
