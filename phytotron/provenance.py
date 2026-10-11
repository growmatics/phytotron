"""Provenance: what a run was made with, so every number traces to one named run.

The run record (commit, config hash, input-file hashes, package versions, arm settings,
hardware), the resume check that refuses a run directory launched with other code, and
the hashes and stamps they are built from.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any

from phytotron.arms import incumbent_params, mpc_settings
from phytotron.config import BenchmarkConfig
from phytotron.controllers.registry import ArmSpec, record_specs
from phytotron.grid import Task
from phytotron.reproducibility import thread_settings
from phytotron.scenarios import GLG2_COMMIT, glg2_config_dir
from phytotron.season import Paths

PACKAGES = ("phytotron", "gl-gym", "numpy", "scipy", "casadi")


def package_versions() -> dict[str, str]:
    """Installed versions of the packages a result depends on, plus Python."""
    versions = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_hashes(cfg: BenchmarkConfig, paths: Paths, years: Sequence[int]) -> dict[str, str]:
    """SHA-256 of every input file a season reads: the simulator's scenario and incumbent
    configs, and the weather file of each year in the grid."""
    config_dir = Path(paths.glg2_config_dir)
    files = {
        "glg2/envs/TomatoEnv.yml": config_dir / "TomatoEnv.yml",
        "glg2/agents/rule_based.yml": config_dir.parent / "agents" / "rule_based.yml",
    }
    location = cfg.generalisation.location
    for year in sorted(set(years)):
        files[f"weather/{location}/{year}.csv"] = Path(paths.weather_dir) / location / f"{year}.csv"
    return {name: file_sha256(path) for name, path in files.items()}


def arm_settings(
    cfg: BenchmarkConfig, paths: Paths, registered: Mapping[str, ArmSpec] | None = None
) -> dict[str, Any]:
    """The parameters each arm runs with, as plain data (registered arms: their spec)."""
    s = mpc_settings(cfg)
    assert s.economic is not None and s.climate_bounds is not None  # from_config sets both
    settings: dict[str, Any] = {
        "incumbent": incumbent_params(paths.glg2_config_dir),
        "physics_mpc": {
            "horizon": s.horizon,
            "dt": s.dt,
            "objective": s.objective,
            "economic": asdict(s.economic),
            "climate_bounds": {k: list(v) for k, v in s.climate_bounds.items()},
            "terminal_weight": cfg.mpc.terminal_weight if cfg.mpc is not None else 0.0,
        },
    }
    for name, spec in (registered or {}).items():
        settings[name] = asdict(spec)
    return settings


def hardware_stamp() -> dict[str, Any]:
    """Machine identity for the run record, so the online-cost axis is interpretable.

    Decision time (the summary key ``online_cost_seconds``) is wall-clock time
    (``time.perf_counter``), so it is **relative**: comparable across arms on the *same*
    machine (the Pareto ordering is valid), but the
    absolute ms/step is not portable across hardware. Stamping the machine makes the
    numbers reproducible in context. A fully hardware-independent cost proxy (IPOPT
    iteration count / NLP evaluations per step) is a possible future refinement.

    The math-library thread settings are recorded too: MPC results depend on them
    (see :mod:`phytotron.reproducibility`).
    """
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "threads": thread_settings(),
    }


def git_output(repo: Path, *args: str) -> str:
    """``git <args>`` in ``repo``, or ``unknown`` when there is no git or no repository."""
    try:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def simulator_commit(config_dir: Path) -> str:
    """The simulator commit: the pinned one for the shipped scenario files, else the checkout's."""
    if Path(config_dir).resolve() == glg2_config_dir().resolve():
        return GLG2_COMMIT[:7]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config_dir, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_record(
    cfg: BenchmarkConfig,
    tasks: list[Task],
    paths: Paths,
    registered: Mapping[str, ArmSpec] | None = None,
    *,
    config: Path,
    arms: list[str],
    workers: int,
) -> dict[str, Any]:
    """``run_record.json``: what a run was launched with, so a number traces to one run.

    The commit and its dirty flag, the config hash, the simulator commit, the time, the
    package versions, the hash of every input file, each arm's settings, the registered
    arms by file hash, the grid, and the hardware (decision time is wall-clock).
    """
    full_days = cfg.generalisation.season_length_days
    repo = config.parent.parent
    return {
        "git_commit": git_output(repo, "rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git_output(repo, "status", "--porcelain")),
        "config_sha256": file_sha256(config),
        "testbed": cfg.testbed["release"],
        "glg2_checkout_commit": simulator_commit(Path(paths.glg2_config_dir)),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "timezone": time.strftime("%Z (UTC%z)"),
        "workers": workers,
        "packages": package_versions(),
        "inputs_sha256": input_hashes(cfg, paths, sorted({t.year for t in tasks})),
        "arm_settings": arm_settings(cfg, paths, registered),
        "custom_arms": record_specs(registered or {}),
        "season_days": tasks[0].season_days if tasks else full_days,
        "smoke": bool(tasks) and tasks[0].season_days < full_days,
        "grid": {
            "arms": arms,
            "years": sorted({t.year for t in tasks}),
            "mismatch_levels": sorted({t.mismatch for t in tasks}),
            "seeds": sorted({t.seed for t in tasks}),
            "seasons": len(tasks),
        },
        "hardware": hardware_stamp(),
    }


class ResumeRefused(RuntimeError):
    """A resume whose code, config or registered arms differ from the launch."""


def check_resume(
    launch: Mapping[str, Any], config: Path, registered: Mapping[str, ArmSpec] | None = None
) -> str | None:
    """Raise :class:`ResumeRefused` if the commit, the config hash or a registered arm's
    file hash differs from the launch record. Returns a warning to print when the working
    tree is dirty, else None."""
    repo = config.parent.parent
    now = {
        "git_commit": git_output(repo, "rev-parse", "--short", "HEAD"),
        "config_sha256": file_sha256(config),
    }
    changed = [k for k, v in now.items() if launch.get(k) != v]
    if changed:
        raise ResumeRefused(
            f"This run was launched with a different {' and '.join(changed)} "
            f"({launch.get('git_commit')}). Resume it from that commit, or use a new --run-dir."
        )
    launched = launch.get("custom_arms") or {}
    now_registered = record_specs(registered or {})
    if {k: v["sha256"] for k, v in now_registered.items()} != {
        k: v.get("sha256") for k, v in launched.items()
    }:
        raise ResumeRefused(
            "This run was launched with different user controllers "
            f"({sorted(launched)}, by file hash). Pass the same --controller specs "
            "with unchanged code, or use a new --run-dir."
        )
    if git_output(repo, "status", "--porcelain"):
        return "Warning: the working tree has uncommitted changes."
    return None
