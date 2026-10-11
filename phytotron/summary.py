"""Summarise a run's season records: per-arm results, gates, safety, Pareto front, rows.

The gate protocol (``gate.verdict``): a model-based arm's verdict pools its seasons with
mismatch above 0; mismatch 0 is the labelled inverse-crime reference and never part of
the verdict. Each mismatch level is also gated on its own. A model-free arm's verdict
covers all its seasons. :func:`rescore_records` keeps only the seasons the current grid
rules would run, so an older run can be scored under new rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np

from phytotron.benchmark import BenchmarkResult, GateCount
from phytotron.config import BenchmarkConfig
from phytotron.controllers.registry import ArmSpec, specs_from_record
from phytotron.envs.glg2 import (
    SeasonResult,
)
from phytotron.grid import Task, is_model_based, seed_invariant
from phytotron.metrics.crop_rules import broken_rules, humidity_deficit
from phytotron.metrics.safety import banded_violation
from phytotron.records import BREAKDOWN_FIELDS, FailureRecord, Record, SeasonRecord, task_json


def season_row(record: SeasonRecord, cfg: BenchmarkConfig) -> dict[str, Any]:
    """One scalar summary row per season: the numbers the analysis needs, no traces.

    The banded flag is given for gated classes only, each with its own depth and unit.
    ``broken_rules`` lists the crop rules each gated class breaks (None without climate
    traces).
    ``fruit_growth_kg_fw`` is revenue divided by the fruit price: the growth of fruit
    fresh weight on the plants over the season, in kg/m2 (not harvested yield).
    """
    step_hours = cfg.episode.step_hours
    task = record.task
    exc = {k: np.asarray(v, dtype=float) for k, v in record.exceedances.items()}
    revenue = record.revenue
    price = cfg.economic.fruit_price if cfg.economic is not None else None
    return {
        "arm": task.arm,
        "label": record.label,
        "seed": task.seed,
        "mismatch": float(task.mismatch),
        "net": record.net_economic_value,
        **{name: getattr(record, name) for name in (*BREAKDOWN_FIELDS, "steps")},
        "fruit_growth_kg_fw": None if revenue is None or not price else revenue / price,
        "online_cost_seconds": record.online_cost_seconds,
        "solve_success": record.solve_success,
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


def _broken(record: SeasonRecord, cfg: BenchmarkConfig) -> dict[str, list[str]] | None:
    climate = record.climate
    if climate is None:
        return None
    step_seconds = cfg.episode.timestep_seconds
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


def night_humidity_deficit(record: SeasonRecord) -> float | None:
    """Mean humidity deficit (g/m3) over the night steps (no outdoor radiation).

    Reported next to the humidity rules because growers steer on it; not gated.
    """
    climate = record.climate
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


def _gate_block(records: list[Record], cfg: BenchmarkConfig, arm: str) -> dict[str, Any] | None:
    """Crop-rule, strict and banded gate for one arm over one subset of its records.

    ``gate_crop`` is None when any completed season lacks climate traces.
    """
    mine = [r for r in records if r.task.arm == arm]
    if not mine:
        return None
    result = _result(mine, cfg)
    bands = {c: cfg.gate.band(c) for c in cfg.gate.soft_classes}
    rules = {c: cfg.gate.rules(c) for c in cfg.gate.soft_classes}
    step_seconds = cfg.episode.timestep_seconds
    mean, std = result.net_value_summary(arm)
    crop = result.crop_gate(arm, step_seconds, rules)
    return {
        "n_seasons": len(mine) - result.not_started.get(arm, 0),
        "net_value_mean": mean,
        "net_value_std": std,
        "gate": _passes(result.strict_gate(arm)),
        "gate_crop": None if crop is None else _passes(crop),
        "gate_crop_counts": None
        if crop is None
        else {
            cls: {"failing": g.failing, "seasons": g.seasons, "upper_bound": g.upper_bound}
            for cls, g in crop.items()
        },
        "gate_banded": _passes(result.banded_gate(arm, bands)),
        "underpowered": result.gate_underpowered(arm),
    }


def _passes(gate: dict[str, GateCount]) -> dict[str, bool]:
    return {cls: g.passes for cls, g in gate.items()}


def _result(records: list[Record], cfg: BenchmarkConfig) -> BenchmarkResult:
    seasons: dict[tuple[str, str, int], SeasonResult] = {}
    failed: dict[str, int] = {}
    not_started: dict[str, int] = {}
    for record in records:
        arm = record.task.arm
        if isinstance(record, SeasonRecord):
            seasons[(arm, record.label, record.task.seed)] = record.result()
        elif record.counted:
            failed[arm] = failed.get(arm, 0) + 1
        else:
            not_started[arm] = not_started.get(arm, 0) + 1
    return BenchmarkResult(
        seasons=seasons,
        gate=cfg.gate,
        soft_classes=cfg.gate.soft_classes,
        step_hours=cfg.episode.step_hours,
        failed=failed,
        not_started=not_started,
    )


def _select(
    records: list[Record],
    arm: str,
    *,
    mismatch: float | None = None,
    above_zero: bool = False,
) -> list[Record]:
    """An arm's records, optionally at one mismatch level or above mismatch 0 only."""
    return [
        r
        for r in records
        if r.task.arm == arm
        and (mismatch is None or r.task.mismatch == mismatch)
        and (not above_zero or r.task.mismatch > 0.0)
    ]


def summarize(
    records: Mapping[str, Record], cfg: BenchmarkConfig, run_record: dict[str, Any]
) -> dict[str, Any]:
    """Per-arm results (net value, gates, safety, cost, Pareto) plus one row per season.

    Per arm: the gate ``verdict`` (see the module docstring), the gate per mismatch level,
    net value and its breakdown, safety profiles, decision time and solve success.
    """
    every = list(records.values())
    result = _result(every, cfg)
    failures = [
        {
            "label": r.label,
            **task_json(r.task, r.task_keys),
            "error": r.error,
            "stage": r.stage,
            "counted": r.counted,
        }
        for r in every
        if isinstance(r, FailureRecord)
    ]
    registered = specs_from_record(run_record)
    arms: dict[str, Any] = {}
    for arm in result.arms():
        model_based = is_model_based(arm, registered)
        verdict_records = _select(every, arm, above_zero=model_based)
        levels = sorted({r.task.mismatch for r in every if r.task.arm == arm})
        completed = [r for (a, _l, _s), r in result.seasons.items() if a == arm]
        mean, std = result.net_value_summary(arm)
        rows = [
            season_row(r, cfg) for r in every if r.task.arm == arm and isinstance(r, SeasonRecord)
        ]
        arms[arm] = {
            "n_seasons": len(completed),
            "n_failed": result.failed.get(arm, 0),
            "seasons_not_started": result.not_started.get(arm, 0),
            "net_value_mean": mean,
            "net_value_std": std,
            "breakdown_mean": {
                name: _mean([row[name] for row in rows])
                for name in (*BREAKDOWN_FIELDS, "fruit_growth_kg_fw")
            },
            "online_cost_seconds": result.decision_time(arm),
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
        "pareto": _pareto(every, cfg, arms, registered),
        "seasons": sorted(
            (season_row(r, cfg) for r in every if isinstance(r, SeasonRecord)),
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
    records: list[Record],
    cfg: BenchmarkConfig,
    arms: dict[str, Any],
    registered: Mapping[str, ArmSpec] | None = None,
) -> dict[str, Any]:
    """Pareto front over the verdict seasons of each arm, gated by the primary gate.

    ``gate.primary`` names the per-season rule whose verdict decides which arms are
    ranked (``crop_rules`` or ``banded``); the strict gate saturates (docs/kpis.md). An
    arm whose primary gate cannot be computed (no climate traces) is not a passer. The
    incumbent is always included as the reference.
    """
    key = "gate_crop" if cfg.gate.primary == "crop_rules" else "gate_banded"
    verdict_records = [
        r for r in records if not is_model_based(r.task.arm, registered) or r.task.mismatch > 0.0
    ]
    passers = [
        a for a, s in arms.items() if s["verdict"].get(key) and all(s["verdict"][key].values())
    ]
    ranked = set(passers) | ({"incumbent"} & set(arms))
    gated = _result([r for r in verdict_records if r.task.arm in ranked], cfg)
    return {
        "primary_gate": cfg.gate.primary,
        "gate_passers": sorted(passers),
        "gated": gated.pareto(incumbent="incumbent", gate_passers_only=False),
        "ungated": _result(verdict_records, cfg).pareto(
            incumbent="incumbent", gate_passers_only=False
        ),
    }


def rescore_records(
    records: Mapping[str, Record],
    cfg: BenchmarkConfig,
    registered: Mapping[str, ArmSpec] | None = None,
) -> dict[str, Record]:
    """Keep only records that the current grid rules would run.

    Applied to a run made under older rules, this drops the seasons that are now known
    duplicates (extra seeds of a seed-invariant season), so they are not counted twice.
    """
    return {
        key: r
        for key, r in records.items()
        if r.task.seed == min(_seeds(records, r.task))
        or not seed_invariant(r.task.arm, float(r.task.mismatch), registered)
    }


def _seeds(records: Mapping[str, Record], task: Task) -> list[int]:
    return [
        r.task.seed
        for r in records.values()
        if r.task.arm == task.arm and r.task.year == task.year and r.task.mismatch == task.mismatch
    ]
