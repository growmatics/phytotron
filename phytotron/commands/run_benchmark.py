"""Run the benchmark grid: every arm on every held-out season, gated and ranked.

Runs the incumbent and the economic MPC over the held-out weather years, with the MPC
at every level of the mismatch ladder and every seed (``configs/benchmark.yaml``). Each
finished season is written to ``<run-dir>/seasons/``, so rerunning the same command
resumes an interrupted run, from the same commit and config only. A failed season stays
failed on resume; ``--retry-failed`` reruns it and keeps the failure in its attempt
history. When the grid is complete it writes ``<run-dir>/summary.json`` and prints the
results.

Every season is checked: it must run all its steps, and every action must stay in its
control kind's range. A season that fails a check counts as failed. ``run_record.json``
stores the commit, package versions, input-file hashes and each arm's settings.

    GLG2_WEATHER_DIR=weather python examples/run_benchmark.py --run-dir runs/full --workers 18

Your own controller, next to the incumbent (see ``docs/own-controller.md``)::

    phytotron run --run-dir runs/mine --controller thermostat=my_controller:Thermostat \
        --arms incumbent thermostat --years 2025 --seeds 0

A quick smoke run (one-day seasons, two mismatch levels, one seed):

    phytotron run --run-dir runs/smoke --season-days 1 \
        --mismatch 0 0.1 --seeds 0 --workers 4

A full MPC season takes about 20 to 25 minutes on one core; the full grid is about 120
MPC seasons. Every worker pins its math libraries to one thread, which MPC results need
to be reproducible (see ``docs/mpc.md``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from phytotron.arms import ARMS
from phytotron.config import default_config_path, load_config
from phytotron.controllers.registry import ArmSpec, parse_specs, specs_from_record
from phytotron.envs.glg2 import paths_from_env
from phytotron.grid import Task, build_tasks
from phytotron.provenance import ResumeRefused, check_resume, git_output, run_record
from phytotron.records import (
    FailureRecord,
    Record,
    load_records,
    pending,
    save_record,
    with_attempts,
)
from phytotron.season import Paths, run_task_safe
from phytotron.summary import rescore_records, summarize


def _verdicts(gate: dict[str, bool] | None) -> str:
    return "n/a" if gate is None else "/".join("pass" if v else "fail" for v in gate.values())


def _num(value: float | None, spec: str) -> str:
    """A number for the printed summary, or ``n/a`` when the summary has none."""
    return "  n/a" if value is None else format(value, spec)


def _gates(block: dict[str, Any]) -> str:
    flag = " (underpowered)" if block["underpowered"] else ""
    return (
        f"crop rules {_verdicts(block.get('gate_crop'))}, strict {_verdicts(block['gate'])}, "
        f"banded {_verdicts(block['gate_banded'])}{flag}"
    )


def _print_summary(summary: dict[str, Any]) -> None:
    rec, protocol = summary["record"], summary["protocol"]
    print(f"\nSimulated benchmark, {rec['season_days']}-day seasons, commit {rec['git_commit']}")
    if rec["smoke"]:
        print("SMOKE RUN: shortened seasons, numbers are not benchmark results.")
    print(f"Gated classes: {', '.join(protocol['gated_classes'])} (results in that order)")
    print(f"Primary gate: {protocol['primary_gate']}; crop-rule numbers {protocol['rules_status']}")
    print(f"Protocol version {protocol.get('version', 1)} (the rules this summary is scored under)")
    if protocol["rules_status"] == "provisional":
        print("PROVISIONAL crop rules: numbers not yet fixed by the horticulturist.")
    for arm, a in summary["arms"].items():
        v = a["verdict"]
        ss = "  -  " if a["solve_success"] is None else f"{a['solve_success']:.3f}"
        cost = (a["online_cost_seconds"] or 0.0) * 1e3
        not_started = a.get("seasons_not_started", 0)
        unstarted = f", {not_started} not started" if not_started else ""
        print(f"\n  {arm}: {a['n_seasons']} seasons, {a['n_failed']} failed{unstarted}, "
              f"{cost:.1f} ms/step, solve success {ss}")  # fmt: skip
        if "gate" in v:
            print(f"    verdict ({v['seasons']}, n={v['n_seasons']}): "
                  f"net {_num(v['net_value_mean'], '+.3f')} +/- "
                  f"{_num(v['net_value_std'], '.3f')} EUR/m2; {_gates(v)}")  # fmt: skip
        b = a["breakdown_mean"]
        if b["revenue"] is not None:
            money = ", ".join(
                f"{label} {b[key]:.3f}"
                for label, key in (
                    ("revenue", "revenue"),
                    ("heating", "heating_cost"),
                    ("CO2", "co2_cost"),
                    ("electricity", "electricity_cost"),
                )
            )
            print(f"    mean per season: {money} EUR/m2; "
                  f"fruit growth {b['fruit_growth_kg_fw']:.2f} kg/m2")  # fmt: skip
        for level, block in a["by_mismatch"].items():
            if block is not None and len(a["by_mismatch"]) > 1:
                print(f"    mismatch {level} (n={block['n_seasons']}): "
                      f"net {_num(block['net_value_mean'], '+.3f')}; {_gates(block)}")  # fmt: skip
    pareto = summary["pareto"]
    print(f"\n  Gate passers ({pareto['primary_gate']}): {pareto['gate_passers'] or 'none'}")
    print(f"  Pareto, gate passers plus the incumbent: {pareto['gated']['non_dominated']}")


def _write_summary(run_dir: Path, name: str, records: dict[str, Any], cfg: Any) -> int:
    summary = summarize(records, cfg, json.loads((run_dir / "run_record.json").read_text()))
    (run_dir / name).write_text(json.dumps(summary, indent=1, allow_nan=False))
    _print_summary(summary)
    print(f"\nWrote {run_dir / name}")
    return 1 if summary["record"]["n_failed_seasons"] else 0


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """The ``run`` command's arguments, on ``parser`` or on a new one."""
    parser = parser or argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="e.g. runs/full")
    parser.add_argument(
        "--config", type=Path, default=default_config_path(), help="configs/benchmark.yaml"
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help=f"arms to run; default: {list(ARMS)} plus every --controller",
    )
    parser.add_argument(
        "--controller",
        action="append",
        default=None,
        metavar="NAME=MODULE:ATTR",
        help="register a user controller as an arm (a class with a no-argument constructor, "
        "or a factory (env, cfg, mismatch, seed) -> controller); repeatable",
    )
    parser.add_argument(
        "--model-based",
        nargs="+",
        default=None,
        metavar="NAME",
        help="user controllers that plan with a model: they run the mismatch ladder and "
        "their verdict pools the seasons above mismatch 0",
    )
    parser.add_argument(
        "--seed-sensitive",
        nargs="+",
        default=None,
        metavar="NAME",
        help="user controllers whose seasons depend on the seed: every seed runs",
    )
    parser.add_argument("--years", nargs="+", type=int, help="default: held-out years")
    parser.add_argument("--mismatch", nargs="+", type=float, help="default: config ladder")
    parser.add_argument("--seeds", nargs="+", type=int, help="default: config seed count")
    parser.add_argument("--season-days", type=int, help="default: config season length")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="rerun failed seasons (kept in each record's attempt history); default: keep them",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="do not run; score the seasons already in --run-dir under the current rules",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


def registered_arms(args: argparse.Namespace) -> dict[str, ArmSpec]:
    """The user controllers from the command line, or a clean error."""
    try:
        return parse_specs(
            args.controller,
            model_based=args.model_based,
            seed_sensitive=args.seed_sensitive,
            reserved=ARMS,
        )
    except ValueError as err:
        sys.exit(str(err))


def run(
    args: argparse.Namespace,
    *,
    season_runner: Callable[[Task, Paths], Record] | None = None,
    executor: Any = None,
) -> int:
    """The ``run`` command. ``season_runner`` and ``executor`` exist for the tests: the
    defaults are :func:`run_task_safe` and a process pool."""
    season_runner = run_task_safe if season_runner is None else season_runner
    executor = ProcessPoolExecutor if executor is None else executor
    cfg = load_config(args.config)
    run_dir: Path = args.run_dir
    if args.rescore:
        launch = json.loads((run_dir / "run_record.json").read_text())
        records = rescore_records(load_records(run_dir), cfg, specs_from_record(launch))
        commit = git_output(args.config.parent.parent, "rev-parse", "--short", "HEAD")
        return _write_summary(run_dir, f"summary-rescored-{commit}.json", records, cfg)
    registered = registered_arms(args)
    args.arms = args.arms or [*ARMS, *registered]

    config_dir, weather_dir = paths_from_env(os.environ)
    paths = Paths(str(args.config), str(config_dir), str(weather_dir))
    try:
        tasks = build_tasks(
            cfg,
            args.arms,
            years=args.years,
            mismatch_levels=args.mismatch,
            seeds=args.seeds,
            season_days=args.season_days,
            registered=registered,
        )
    except ValueError as err:
        sys.exit(str(err))
    run_dir.mkdir(parents=True, exist_ok=True)
    grid_file = run_dir / "grid.json"
    grid = [t.key for t in tasks]
    if grid_file.exists() and json.loads(grid_file.read_text()) != grid:
        sys.exit(f"{run_dir} holds a different grid; use a new --run-dir.")
    grid_file.write_text(json.dumps(grid, indent=1))
    record_file = run_dir / "run_record.json"
    if record_file.exists():  # the launch record survives resumes
        try:
            warning = check_resume(json.loads(record_file.read_text()), args.config, registered)
        except ResumeRefused as err:
            sys.exit(str(err))
        if warning:
            print(warning)
    else:
        record = run_record(
            cfg, tasks, paths, registered, config=args.config, arms=args.arms, workers=args.workers
        )
        record_file.write_text(json.dumps(record, indent=2))

    done = load_records(run_dir)
    todo = pending(tasks, done, retry_failed=args.retry_failed)
    # Longest seasons first (MPC before the incumbent), so the pool drains evenly.
    todo.sort(key=lambda t: t.arm != "physics_mpc")
    print(f"{len(tasks)} seasons in the grid, {len(todo)} to run, {args.workers} workers")
    start = time.time()
    with executor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(season_runner, t, paths): t for t in todo}
        for count, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            outcome = with_attempts(future.result(), done.get(task.key))
            save_record(run_dir, task, outcome)
            status = (
                f"FAILED at {outcome.stage}: {outcome.error}"
                if isinstance(outcome, FailureRecord)
                else f"net {outcome.net_economic_value:+.4f}"
            )
            minutes = (time.time() - start) / 60
            print(f"  [{count}/{len(todo)}] {task.arm} {task.label} s{task.seed}: {status}"
                  f"  ({minutes:.0f} min)", flush=True)  # fmt: skip

    records = {k: v for k, v in load_records(run_dir).items() if k in set(grid)}
    return _write_summary(run_dir, "summary.json", records, cfg)
