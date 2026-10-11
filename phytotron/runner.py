"""The benchmark grid: which seasons to run, how to run one, and how to summarise them.

A full grid takes hours, so this module splits it into plain-data :class:`Task` objects
that worker processes can run independently, and that a run directory can store one file
each. An interrupted run then resumes where it stopped, and a crashed season is recorded
as a failure instead of aborting the grid. :mod:`phytotron.benchmark` holds the
aggregation (the gates, the safety profiles, the Pareto front) a summary is built from.

Axes per arm, from ``configs/benchmark.yaml``:

- every arm runs every held-out weather year;
- only model-based arms (``physics_mpc``) sweep the mismatch ladder, since the others
  have no model to perturb;
- a season runs one seed only when seeds cannot change it: a deterministic arm (the
  incumbent) on this deterministic simulator, or a model-based arm at mismatch 0, whose
  nominal model is then the plant's own for every seed. More seeds would add identical
  seasons, not independent trials.

The gate protocol (``gate.verdict``): a model-based arm's verdict pools its seasons with
mismatch above 0; mismatch 0 is the labelled inverse-crime reference and never part of
the verdict. Each mismatch level is also gated on its own. A model-free arm's verdict
covers all its seasons.

``phytotron run`` (:mod:`phytotron.commands.run_benchmark`) is the command-line front end.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from phytotron.arms import (
    ARMS,
    DETERMINISTIC_ARMS,
    MODEL_BASED_ARMS,
    build_controller,
    incumbent_params,
    mpc_settings,
)
from phytotron.benchmark import BenchmarkResult
from phytotron.config import BenchmarkConfig, load_config
from phytotron.controllers.registry import ArmSpec, build_custom, specs_from_record
from phytotron.envs.glg2 import (
    SeasonResult,
    expected_steps,
    make_benchmark_env,
    run_season_controller,
)
from phytotron.metrics.crop_rules import broken_rules, humidity_deficit
from phytotron.metrics.safety import banded_violation

_BREAKDOWN = ("revenue", "heating_cost", "co2_cost", "electricity_cost")
_EXTRAS = ("steps", "solve_counts", "max_action_excess")  # optional in older records
STAGES = ("setup", "build", "season", "check")  # where a failed season stopped


class SeasonCheckError(RuntimeError):
    """A season ran but failed a runner check (too few steps, an action out of range)."""


@dataclass(frozen=True)
class Task:
    """One season to run: an arm on one weather year, mismatch level and seed."""

    arm: str
    year: int
    mismatch: float
    seed: int
    season_days: int
    controller: str | None = None  # "module:attr" for a user controller (see registry)

    @property
    def label(self) -> str:
        """Condition label, e.g. ``2025/m0.10`` (the format of the recorded runs)."""
        return f"{self.year}/m{self.mismatch:.2f}"

    @property
    def key(self) -> str:
        """File-safe identifier, unique within one run. Keep this format stable: run
        directories written by earlier versions are read back by it."""
        return f"{self.arm}__{self.year}__m{self.mismatch:.3f}__s{self.seed}"


@dataclass(frozen=True)
class Paths:
    """Where a worker finds the benchmark config, GreenLight-Gym2's configs, and weather."""

    benchmark_config: str
    glg2_config_dir: str
    weather_dir: str


def seed_invariant(arm: str, mismatch: float, custom: Mapping[str, ArmSpec] | None = None) -> bool:
    """True when the seed cannot change the season, so one seed is enough."""
    if custom and arm in custom:
        return custom[arm].seed_invariant(mismatch)
    return arm in DETERMINISTIC_ARMS or (arm in MODEL_BASED_ARMS and mismatch == 0.0)


def is_model_based(arm: str, custom: Mapping[str, ArmSpec] | None = None) -> bool:
    """Model-based arms sweep the mismatch ladder; their verdict pools mismatch above 0."""
    if custom and arm in custom:
        return custom[arm].model_based
    return arm in MODEL_BASED_ARMS


def build_tasks(
    cfg: BenchmarkConfig,
    arms: Sequence[str] = ARMS,
    *,
    years: Sequence[int] | None = None,
    mismatch_levels: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
    season_days: int | None = None,
    custom: Mapping[str, ArmSpec] | None = None,
) -> list[Task]:
    """The grid for ``arms``, with config defaults for every axis not given.

    ``custom`` maps the names of user controllers (``--controller``) to their specs; a
    custom arm's tasks carry its ``module:attr`` target so workers can build it.
    """
    custom = dict(custom or {})
    known = [*ARMS, *custom]
    unknown = sorted(set(arms) - set(known))
    if unknown:
        raise ValueError(f"unknown arms {unknown}; available: {known}")
    years = list(years or cfg.generalisation.heldout_years)
    levels = list(mismatch_levels or cfg.raw["mismatch"]["sweep_magnitudes"])
    seeds = list(seeds or range(int(cfg.raw["reporting"]["seeds_per_condition"])))
    days = season_days or cfg.generalisation.season_length_days
    tasks = []
    for arm in arms:
        arm_levels = levels if is_model_based(arm, custom) else [0.0]
        target = custom[arm].target if arm in custom else None
        for year in years:
            for level in arm_levels:
                level_seeds = seeds[:1] if seed_invariant(arm, float(level), custom) else seeds
                for seed in level_seeds:
                    tasks.append(Task(arm, int(year), float(level), int(seed), int(days), target))
    return tasks


def run_task(task: Task, paths: Paths, progress: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one season and return it as a JSON-ready record, checked.

    ``progress["stage"]`` follows the run, so a caller can tell where a failure happened:
    ``setup`` (config, environment, reset: the harness's), ``build`` (the controller's
    constructor), ``season`` and ``check``. A season that fails a check raises
    :class:`SeasonCheckError` with the record attached.
    """
    progress = {} if progress is None else progress
    progress["stage"] = "setup"
    started, t0 = _utc_now(), time.perf_counter()
    cfg = load_config(paths.benchmark_config)
    env = make_benchmark_env(
        cfg,
        task.year,
        config_dir=paths.glg2_config_dir,
        weather_dir=paths.weather_dir,
        season_days=task.season_days,
    )
    env.reset(seed=task.seed)
    progress["stage"] = "build"
    controller = (
        build_custom(task.controller, env, cfg, mismatch=task.mismatch, seed=task.seed)
        if task.controller
        else build_controller(
            task.arm,
            env,
            cfg,
            glg2_config_dir=paths.glg2_config_dir,
            mismatch=task.mismatch,
            seed=task.seed,
        )
    )
    progress["stage"] = "season"
    record = season_record(task, run_season_controller(env, controller, task.seed))
    record.update(started=started, finished=_utc_now(), wall_seconds=time.perf_counter() - t0)
    progress["stage"] = "check"
    problems = season_problems(record, cfg)
    if problems:
        err = SeasonCheckError("; ".join(problems))
        err.record = record  # type: ignore[attr-defined]
        raise err
    return record


def season_problems(record: dict[str, Any], cfg: BenchmarkConfig) -> list[str]:
    """Why a finished season cannot be scored as a normal season (empty if it can).

    Too few steps means the simulator ended the season early (GreenLight-Gym2's gym step
    ends a season on an ODE error without raising). An action outside its range means the
    plant received controls it should not have (``step_raw_control`` does not clip).
    """
    problems = []
    step_seconds = int(cfg.raw["episode"]["timestep_seconds"])
    want = expected_steps(int(record["task"]["season_days"]), step_seconds)
    if record.get("steps") is not None and record["steps"] != want:
        problems.append(f"ran {record['steps']} of {want} steps")
    excess = record.get("max_action_excess")
    tolerance = float(cfg.raw["checks"]["action_tolerance"])
    if excess is not None and excess > tolerance:
        problems.append(f"an action left its range by {excess:.3g}")
    return problems


def run_task_safe(task: Task, paths: Paths) -> dict[str, Any]:
    """Like :func:`run_task`, but a season that raises becomes an error record.

    A crashed season must be recorded, not dropped: the record says at which stage it
    failed (see :func:`counted_failure` for which stages count against the arm's gate),
    and a season that failed a check keeps its data. (A worker process that dies outright still
    stops the pool; rerunning the same command resumes the grid.)
    """
    progress: dict[str, Any] = {}
    try:
        return run_task(task, paths, progress)
    except Exception as exc:  # deliberately broad: isolate one season's failure
        return {
            **getattr(exc, "record", {"task": asdict(task), "label": task.label}),
            "error": f"{type(exc).__name__}: {exc}".splitlines()[0][:200],
            "stage": progress.get("stage", "setup"),
            "failed_at": _utc_now(),
            "traceback": traceback.format_exc(),
        }


def with_attempts(record: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    """``record`` with the attempt history of the record it replaces (a failed earlier run).

    Each failed attempt is kept as its error, stage and time, so a retried season shows
    that it failed before.
    """
    history = list((previous or {}).get("attempts", []))
    if previous is not None and "error" in previous:
        history.append({k: previous.get(k) for k in ("error", "stage", "failed_at")})
    return {**record, "attempts": history}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def season_record(task: Task, result: SeasonResult) -> dict[str, Any]:
    """A season result as plain data, with the full per-step exceedance traces."""
    return {
        "task": asdict(task),
        "label": task.label,
        "net_economic_value": float(result.net_economic_value),
        "online_cost_seconds": float(result.online_cost_seconds),
        "solve_success": result.solve_success,
        **{name: getattr(result, name) for name in (*_BREAKDOWN, *_EXTRAS)},
        "exceedances": {
            k: np.asarray(v, dtype=float).tolist() for k, v in result.exceedances.items()
        },
        "climate": None
        if result.climate is None
        else {k: np.asarray(v, dtype=float).round(4).tolist() for k, v in result.climate.items()},
    }


def season_from_record(record: dict[str, Any]) -> SeasonResult:
    """Inverse of :func:`season_record`. Records written before the cost breakdown or the
    climate traces existed give ``None`` for them."""
    climate = record.get("climate")
    return SeasonResult(
        net_economic_value=float(record["net_economic_value"]),
        exceedances={k: np.asarray(v, dtype=float) for k, v in record["exceedances"].items()},
        online_cost_seconds=float(record["online_cost_seconds"]),
        solve_success=record["solve_success"],
        **{name: record.get(name) for name in (*_BREAKDOWN, *_EXTRAS)},
        climate=None
        if climate is None
        else {k: np.asarray(v, dtype=float) for k, v in climate.items()},
    )


def save_record(run_dir: Path, task: Task, record: dict[str, Any]) -> Path:
    """Write one season record to ``run_dir/seasons/<key>.json`` (atomic rename)."""
    path = run_dir / "seasons" / f"{task.key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(path)
    return path


def load_records(run_dir: Path) -> dict[str, dict[str, Any]]:
    """All season records in a run directory, keyed by task key."""
    folder = run_dir / "seasons"
    if not folder.is_dir():
        return {}
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def pending(
    tasks: Sequence[Task], done: dict[str, dict[str, Any]], *, retry_failed: bool = False
) -> list[Task]:
    """Tasks still to run: those without a record, plus failed ones if ``retry_failed``.

    A failed season stays failed on resume unless a retry is asked for, so a resumed
    run cannot quietly replace a crash with a lucky second attempt.
    """
    return [t for t in tasks if t.key not in done or (retry_failed and "error" in done[t.key])]


def season_row(record: dict[str, Any], cfg: BenchmarkConfig) -> dict[str, Any]:
    """One scalar summary row per season: the numbers the analysis needs, no traces.

    The banded flag is given for gated classes only, each with its own depth and unit.
    ``broken_rules`` lists the crop rules each gated class breaks (None without climate
    traces).
    ``fruit_growth_kg_fw`` is revenue divided by the fruit price: the growth of fruit
    fresh weight on the plants over the season, in kg/m2 (not harvested yield).
    """
    step_hours = float(cfg.raw["episode"]["timestep_seconds"]) / 3600.0
    task = record["task"]
    exc = {k: np.asarray(v, dtype=float) for k, v in record["exceedances"].items()}
    revenue = record.get("revenue")
    price = cfg.economic.fruit_price if cfg.economic is not None else None
    return {
        "arm": task["arm"],
        "label": record["label"],
        "seed": task["seed"],
        "mismatch": float(task["mismatch"]),
        "net": record["net_economic_value"],
        **{name: record.get(name) for name in (*_BREAKDOWN, "steps")},
        "fruit_growth_kg_fw": None if revenue is None or not price else revenue / price,
        "online_cost_seconds": record["online_cost_seconds"],
        "solve_success": record["solve_success"],
        "violated": {k: bool(v.max(initial=0.0) > 0.0) for k, v in exc.items()},
        "violated_banded": {
            k: banded_violation(exc[k], step_hours, *cfg.gate.band(k))
            for k in cfg.gate.soft_classes
            if k in exc
        },
        "broken_rules": _broken(record, cfg),
        "night_humidity_deficit_g_m3": night_humidity_deficit(record),
        "max_excursion": {k: float(v.max(initial=0.0)) for k, v in exc.items()},
        "degree_hours": {k: float(v.sum() * step_hours) for k, v in exc.items()},
    }


def _broken(record: dict[str, Any], cfg: BenchmarkConfig) -> dict[str, list[str]] | None:
    climate = record.get("climate")
    if climate is None:
        return None
    step_seconds = int(cfg.raw["episode"]["timestep_seconds"])
    radiation = climate.get("radiation")
    return {
        c: broken_rules(
            np.asarray(climate[c], dtype=float),
            step_seconds,
            cfg.gate.rules(c),
            None if radiation is None else np.asarray(radiation, dtype=float),
        )
        for c in cfg.gate.soft_classes
    }


def night_humidity_deficit(record: dict[str, Any]) -> float | None:
    """Mean humidity deficit (g/m3) over the night steps (no outdoor radiation).

    Reported next to the humidity rules because growers steer on it; not gated.
    """
    climate = record.get("climate")
    if climate is None or climate.get("radiation") is None:
        return None
    night = np.asarray(climate["radiation"], dtype=float) <= 0.0
    if not night.any():
        return None
    deficit = humidity_deficit(
        np.asarray(climate["temperature"], dtype=float),
        np.asarray(climate["humidity"], dtype=float),
    )
    return float(deficit[night].mean())


def _gate_block(
    records: list[dict[str, Any]], cfg: BenchmarkConfig, arm: str
) -> dict[str, Any] | None:
    """Crop-rule, strict and banded gate for one arm over one subset of its records.

    ``gate_crop`` is None when any completed season lacks climate traces.
    """
    mine = [r for r in records if r["task"]["arm"] == arm]
    if not mine:
        return None
    result = _result(mine, cfg)
    bands = {c: cfg.gate.band(c) for c in cfg.gate.soft_classes}
    rules = {c: cfg.gate.rules(c) for c in cfg.gate.soft_classes}
    step_seconds = int(cfg.raw["episode"]["timestep_seconds"])
    mean, std = result.net_value_summary(arm)
    return {
        "n_seasons": len(mine) - result.not_started.get(arm, 0),
        "net_value_mean": mean,
        "net_value_std": std,
        "gate": result.gate_outcome(arm),
        "gate_crop": result.crop_gate_outcome(arm, step_seconds, rules),
        "gate_crop_counts": result.crop_gate_counts(arm, step_seconds, rules),
        "gate_banded": result.banded_gate_outcome(arm, bands),
        "underpowered": result.gate_underpowered(arm),
    }


def counted_failure(record: dict[str, Any]) -> bool:
    """Whether a failed season counts against its controller's gate.

    A season that failed at stage ``setup`` (config, environment, reset) never reached the
    controller: it is listed as not started and not counted. A failure while the
    controller was built, during the season or in its checks is the controller's. A
    failure record without a stage (written before stages existed) counts.
    """
    return record.get("stage") != "setup"


def _result(records: list[dict[str, Any]], cfg: BenchmarkConfig) -> BenchmarkResult:
    seasons: dict[tuple[str, str, int], SeasonResult] = {}
    failed: dict[str, int] = {}
    not_started: dict[str, int] = {}
    for record in records:
        arm = record["task"]["arm"]
        if "error" not in record:
            seasons[(arm, record["label"], record["task"]["seed"])] = season_from_record(record)
        elif counted_failure(record):
            failed[arm] = failed.get(arm, 0) + 1
        else:
            not_started[arm] = not_started.get(arm, 0) + 1
    return BenchmarkResult(
        seasons=seasons,
        record={},
        gate=cfg.gate,
        soft_classes=cfg.gate.soft_classes,
        step_hours=float(cfg.raw["episode"]["timestep_seconds"]) / 3600.0,
        failed=failed,
        not_started=not_started,
    )


def _select(
    records: list[dict[str, Any]],
    arm: str,
    *,
    mismatch: float | None = None,
    above_zero: bool = False,
) -> list[dict[str, Any]]:
    """An arm's records, optionally at one mismatch level or above mismatch 0 only."""
    return [
        r
        for r in records
        if r["task"]["arm"] == arm
        and (mismatch is None or r["task"]["mismatch"] == mismatch)
        and (not above_zero or r["task"]["mismatch"] > 0.0)
    ]


def summarize(
    records: dict[str, dict[str, Any]], cfg: BenchmarkConfig, run_record: dict[str, Any]
) -> dict[str, Any]:
    """Per-arm results (net value, gates, safety, cost, Pareto) plus one row per season.

    Per arm: the gate ``verdict`` (see the module docstring), the gate per mismatch level,
    net value and its breakdown, safety profiles, decision time and solve success.
    """
    every = list(records.values())
    result = _result(every, cfg)
    failures = [
        {
            "label": r["label"],
            **r["task"],
            "error": r["error"],
            "stage": r.get("stage"),
            "counted": counted_failure(r),
        }
        for r in every
        if "error" in r
    ]
    custom = specs_from_record(run_record)
    arms: dict[str, Any] = {}
    for arm in result.arms():
        model_based = is_model_based(arm, custom)
        verdict_records = _select(every, arm, above_zero=model_based)
        levels = sorted({r["task"]["mismatch"] for r in every if r["task"]["arm"] == arm})
        completed = [r for (a, _l, _s), r in result.seasons.items() if a == arm]
        mean, std = result.net_value_summary(arm)
        rows = [season_row(r, cfg) for r in every if r["task"]["arm"] == arm and "error" not in r]
        arms[arm] = {
            "n_seasons": len(completed),
            "n_failed": result.failed.get(arm, 0),
            "seasons_not_started": result.not_started.get(arm, 0),
            "net_value_mean": mean,
            "net_value_std": std,
            "breakdown_mean": {
                name: _mean([row[name] for row in rows])
                for name in (*_BREAKDOWN, "fruit_growth_kg_fw")
            },
            "online_cost_seconds": result.online_cost(arm),
            "solve_success": result.solve_success(arm),
            "verdict": {
                "seasons": "mismatch above 0, pooled" if model_based else "all",
                **(_gate_block(verdict_records, cfg, arm) or {"n_seasons": 0}),
            },
            "by_mismatch": {
                f"{level:.3f}": _gate_block(_select(every, arm, mismatch=level), cfg, arm)
                for level in levels
            },
            "safety": {
                cls: asdict(result.safety(arm, cls))
                for cls in sorted({c for r in completed for c in r.exceedances})
            },
        }
    return {
        "record": {**run_record, "n_failed_seasons": len(failures), "failed_seasons": failures},
        "protocol": {
            "version": cfg.gate.protocol_version,
            "verdict": cfg.gate.verdict,
            "gated_classes": list(cfg.gate.soft_classes),
            "bands": {c: list(cfg.gate.band(c)) for c in cfg.gate.soft_classes},
            "primary_gate": cfg.gate.primary,
            "rules_status": cfg.gate.rules_status,
            "rules": {c: cfg.gate.rules(c) for c in cfg.gate.soft_classes},
        },
        "arms": arms,
        "pareto": _pareto(every, cfg, arms, custom),
        "seasons": sorted(
            (season_row(r, cfg) for r in every if "error" not in r),
            key=lambda r: (r["arm"], r["label"], r["seed"]),
        ),
    }


def _mean(values: list[Any]) -> float | None:
    """The mean of a per-season field; None when no season has it (records written before
    the field existed). A field some seasons have and others lack is a bug, never an
    average over fewer seasons than the summary claims."""
    known = [float(v) for v in values if v is not None]
    if known and len(known) != len(values):
        raise ValueError(f"{len(values) - len(known)} of {len(values)} seasons lack a value")
    return float(np.mean(known)) if known else None


def _pareto(
    records: list[dict[str, Any]],
    cfg: BenchmarkConfig,
    arms: dict[str, Any],
    custom: Mapping[str, ArmSpec] | None = None,
) -> dict[str, Any]:
    """Pareto front over the verdict seasons of each arm, gated by the primary gate.

    ``gate.primary`` names the per-season rule whose verdict decides which arms are
    ranked (``crop_rules`` or ``banded``); the strict gate saturates (docs/kpis.md). An
    arm whose primary gate cannot be computed (no climate traces) is not a passer. The
    incumbent is always included as the reference.
    """
    key = "gate_crop" if cfg.gate.primary == "crop_rules" else "gate_banded"
    verdict_records = [
        r
        for r in records
        if not is_model_based(r["task"]["arm"], custom) or r["task"]["mismatch"] > 0.0
    ]
    passers = [
        a for a, s in arms.items() if s["verdict"].get(key) and all(s["verdict"][key].values())
    ]
    ranked = set(passers) | ({"incumbent"} & set(arms))
    gated = _result([r for r in verdict_records if r["task"]["arm"] in ranked], cfg)
    return {
        "primary_gate": cfg.gate.primary,
        "gate_passers": sorted(passers),
        "gated": gated.pareto(incumbent="incumbent", gate_passers_only=False),
        "ungated": _result(verdict_records, cfg).pareto(
            incumbent="incumbent", gate_passers_only=False
        ),
    }


def rescore_records(
    records: dict[str, dict[str, Any]],
    cfg: BenchmarkConfig,
    custom: Mapping[str, ArmSpec] | None = None,
) -> dict[str, dict[str, Any]]:
    """Keep only records that the current grid rules would run.

    Applied to a run made under older rules, this drops the seasons that are now known
    duplicates (extra seeds of a seed-invariant season), so they are not counted twice.
    """
    return {
        key: r
        for key, r in records.items()
        if r["task"]["seed"] == min(_seeds(records, r["task"]))
        or not seed_invariant(r["task"]["arm"], float(r["task"]["mismatch"]), custom)
    }


def _seeds(records: dict[str, dict[str, Any]], task: dict[str, Any]) -> list[int]:
    return [
        r["task"]["seed"]
        for r in records.values()
        if r["task"]["arm"] == task["arm"]
        and r["task"]["year"] == task["year"]
        and r["task"]["mismatch"] == task["mismatch"]
    ]


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
    cfg: BenchmarkConfig, paths: Paths, custom: Mapping[str, ArmSpec] | None = None
) -> dict[str, Any]:
    """The parameters each arm runs with, as plain data (custom arms: their spec)."""
    mpc = mpc_settings(cfg)
    mpc["economic"] = asdict(mpc["economic"])
    mpc["climate_bounds"] = {k: list(v) for k, v in mpc["climate_bounds"].items()}
    settings: dict[str, Any] = {
        "incumbent": incumbent_params(paths.glg2_config_dir),
        "physics_mpc": mpc,
    }
    for name, spec in (custom or {}).items():
        settings[name] = asdict(spec)
    return settings
