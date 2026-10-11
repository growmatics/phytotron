"""Write ``site/content/results.json``: the results page's numbers, from a run's summary.

Every number on the website's results page comes from ``results/<run>/summary.json``
through this script. The website renders ``results.json`` and never types a number.

    python site/build_results.py          # rewrite results.json
    python site/build_results.py --check  # exit 1 if results.json is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import build_rules

from phytotron.config import load_config

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "2026-10-10-run-2"  # the run the page reports
OUT = ROOT / "site" / "content" / "results.json"

ARM_NAMES = {"incumbent": "Incumbent (rule-based)", "physics_mpc": "Economic MPC"}
RULE_ORDER = ("mean_6d", "mean_24h", "dose", "peak", "humid_hours", "longest_run")
MS = 1000.0


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rule_breaks(seasons: list[dict[str, Any]]) -> dict[str, int]:
    counts = dict.fromkeys(RULE_ORDER, 0)
    for s in seasons:
        for rules in s["broken_rules"].values():
            for rule in rules:
                counts[rule] += 1
    return counts


def _gate_row(arm: str, verdict: dict[str, Any]) -> dict[str, Any]:
    counts = verdict["gate_crop_counts"]
    passed = all(verdict["gate_crop"].values())
    return {
        "arm": arm,
        "name": ARM_NAMES[arm],
        "seasons": verdict["n_seasons"],
        "net_mean": round(verdict["net_value_mean"], 2),
        "net_sd": round(verdict["net_value_std"], 2),
        "classes": {
            cls: {
                "failing": c["failing"],
                "seasons": c["seasons"],
                "upper_bound": _pct(c["upper_bound"]),
            }
            for cls, c in counts.items()
        },
        "gate": "pass" if passed else "fail",
        "underpowered": verdict["underpowered"],
    }


def build(run: Path = RUN, traces: Path | None = None) -> dict[str, Any]:
    """The results document for one run directory.

    ``traces`` is the run's ``seasons/`` directory with the per-season climate traces, not
    part of the repository. Given, the trace-derived facts are recomputed; absent, the
    facts already in ``results.json`` are carried over.
    """
    summary = json.loads((run / "summary.json").read_text())
    record = summary["record"]
    seasons = summary["seasons"]
    arms = summary["arms"]
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in seasons:
        by_arm[s["arm"]].append(s)

    mpc = arms["physics_mpc"]
    mismatch_rows = []
    for level, block in sorted(mpc["by_mismatch"].items(), key=lambda kv: float(kv[0])):
        rows = [s for s in by_arm["physics_mpc"] if f"{s['mismatch']:.3f}" == level]
        mismatch_rows.append(
            {
                "mismatch": _pct(float(level)) if float(level) else "0 (reference)",
                "seasons": block["n_seasons"],
                "net_mean": round(block["net_value_mean"], 2),
                "net_sd": round(block["net_value_std"], 2),
                "fruit_kg_m2": round(_mean([s["fruit_growth_kg_fw"] for s in rows]), 2),
                "solve_success": round(_mean([s["solve_success"] for s in rows]), 3),
                "breaks": _rule_breaks(rows),
            }
        )

    money = []
    for arm in ("incumbent", "physics_mpc"):
        b = arms[arm]["breakdown_mean"]
        money.append(
            {
                "arm": arm,
                "name": ARM_NAMES[arm],
                "seasons": arms[arm]["n_seasons"],
                "revenue": round(b["revenue"], 2),
                "heating": round(b["heating_cost"], 2),
                "co2": round(b["co2_cost"], 2),
                "electricity": round(b["electricity_cost"], 2),
                "net": round(arms[arm]["net_value_mean"], 2),
                "fruit_kg_m2": round(b["fruit_growth_kg_fw"], 2),
                "decision_ms": round(arms[arm]["online_cost_seconds"] * MS, 1),
                "night_humidity_deficit_g_m3": round(
                    _mean([s["night_humidity_deficit_g_m3"] for s in by_arm[arm]]), 2
                ),
                "breaks": _rule_breaks(by_arm[arm]),
                "by_year": {
                    year: round(_mean(nets), 2)
                    for year, nets in sorted(_by_year(by_arm[arm]).items())
                },
            }
        )

    safety = []
    for arm in ("incumbent", "physics_mpc"):
        s = arms[arm]["safety"]
        safety.append(
            {
                "arm": arm,
                "name": ARM_NAMES[arm],
                "temperature_hours": round(s["temperature"]["mean_duration_h"]),
                "temperature_degree_hours": round(s["temperature"]["degree_hours"]),
                "temperature_worst_c": round(s["temperature"]["worst_excursion"], 1),
                "humidity_hours": round(s["humidity"]["mean_duration_h"]),
                "humidity_degree_hours": round(s["humidity"]["degree_hours"]),
                "humidity_worst_points": round(s["humidity"]["worst_excursion"], 1),
                "co2_seasons_with_excursion": round(
                    s["co2"]["season_violation_prob"] * arms[arm]["n_seasons"]
                ),
            }
        )

    rescored = _rescored(run, by_arm)
    facts = trace_facts(traces, load_config(build_rules.CONFIG)) if traces else _carried_facts()
    return {
        "generated_from": {
            "run": str(run.relative_to(ROOT)),
            "phytotron_commit": record["git_commit"],
            "config_sha256": record["config_sha256"],
            "simulator": f"GreenLight-Gym2 {record['testbed']} ({record['glg2_checkout_commit']})",
            "date": record["timestamp"][:10],
            "packages": record["packages"],
            "workers": record["workers"],
            "season_days": record["season_days"],
            "seasons_run": len(seasons),
            "seasons_failed": sum(a["n_failed"] for a in arms.values()),
            "seasons_not_started": sum(a.get("seasons_not_started", 0) for a in arms.values()),
            "recorded_states_per_season": seasons[0]["steps"],  # control steps + the initial state
        },
        "protocol": {
            "version": summary["protocol"].get("version", 1),
            "verdict": summary["protocol"]["verdict"],
            "gated_classes": summary["protocol"]["gated_classes"],
            "rules": summary["protocol"]["rules"],
            "band": summary["protocol"]["bands"],
        },
        "rescored": rescored,
        "trace_facts": facts,
        "gate_passers": summary["pareto"]["gate_passers"],
        "verdicts": [
            _gate_row("incumbent", arms["incumbent"]["verdict"]),
            _gate_row("physics_mpc", mpc["verdict"]),
        ],
        "mpc_by_mismatch": mismatch_rows,
        "money": money,
        "safety": safety,
    }


def trace_facts(seasons_dir: Path, cfg: Any) -> dict[str, Any]:
    """Facts that need the per-season climate traces (held by Growmatics, see the docs).

    The step and the absolute temperature limits come from ``cfg`` (the peak rule), never
    from this file. Failed seasons and records without traces are left out.
    """
    loaded = [json.loads(p.read_text()) for p in sorted(seasons_dir.glob("*.json"))]
    records = [r for r in loaded if "error" not in r and r.get("climate")]
    per_day = 86400 // build_rules.step_seconds(cfg)
    started = [datetime.fromisoformat(r["started"]) for r in records]
    finished = [datetime.fromisoformat(r["finished"]) for r in records]
    mpc = [r for r in records if r["task"]["arm"] == "physics_mpc"]
    incumbent = [r for r in records if r["task"]["arm"] == "incumbent"]
    low, high = build_rules.peak_limits(cfg)
    dips: dict[str, dict[str, Any]] = {}
    for r in incumbent:
        t = r["climate"]["temperature"]
        steps = [i for i, v in enumerate(t) if v < low]
        single = all((i - 1 not in steps) and (i + 1 not in steps) for i in steps)
        dips[r["label"]] = {
            "steps": len(steps),
            "days": [round(i / per_day, 1) for i in steps],
            "all_single_steps": single,
        }
    dip_days = [day for d in dips.values() for day in d["days"]]
    return {
        "source": "the per-season files with climate traces of this run (not in the repository)",
        "wall_minutes": round((max(finished) - min(started)).total_seconds() / 60),
        "mpc_seasons_below_limit": sum(
            any(v < low for v in r["climate"]["temperature"]) for r in mpc
        ),
        "mpc_seasons_below_limit_after_day_1": sum(
            any(v < low for v in r["climate"]["temperature"][per_day:]) for r in mpc
        ),
        "mpc_seasons_above_limit": sum(
            any(v > high for v in r["climate"]["temperature"]) for r in mpc
        ),
        "incumbent_dips_below_limit": dips,
        "incumbent_dips_per_season_min_max": [
            min(d["steps"] for d in dips.values()),
            max(d["steps"] for d in dips.values()),
        ],
        "incumbent_dip_days_min_max": [min(dip_days), max(dip_days)] if dip_days else None,
    }


def _carried_facts() -> dict[str, Any] | None:
    if OUT.exists():
        return json.loads(OUT.read_text()).get("trace_facts")
    return None


def _rescored(run: Path, original: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    """The same run scored under a later protocol (``summary-protocol-<N>.json``), if any.

    Reports, per controller, the rule breaks and the class verdicts under both protocols,
    so a reader sees exactly what the rule change moved.
    """
    files = sorted(run.glob("summary-protocol-*.json"))
    if not files:
        return None
    summary = json.loads(files[-1].read_text())
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in summary["seasons"]:
        by_arm[s["arm"]].append(s)
    arms = []
    for arm in ("incumbent", "physics_mpc"):
        before, after = original[arm], by_arm[arm]
        arms.append(
            {
                "arm": arm,
                "name": ARM_NAMES[arm],
                "seasons": len(after),
                "breaks_before": _rule_breaks(before),
                "breaks_after": _rule_breaks(after),
                "class_failures_before": _class_failures(before),
                "class_failures_after": _class_failures(after),
                "verdict": _gate_row(arm, summary["arms"][arm]["verdict"]),
            }
        )
    return {
        "protocol_version": summary["protocol"]["version"],
        "rules": summary["protocol"]["rules"],
        "file": str(files[-1].relative_to(ROOT)),
        "gate_passers": summary["pareto"]["gate_passers"],
        "arms": arms,
    }


def _class_failures(seasons: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for s in seasons:
        for cls, rules in s["broken_rules"].items():
            counts[cls] += bool(rules)
    return dict(sorted(counts.items()))


def _by_year(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    years: dict[str, list[float]] = defaultdict(list)
    for s in rows:
        years[s["label"].split("/")[0]].append(s["net"])
    return years


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if results.json is stale")
    parser.add_argument(
        "--traces", type=Path, help="the run's seasons/ directory, to recompute the trace facts"
    )
    args = parser.parse_args()
    fresh = render(build(traces=args.traces))
    if args.check:
        if not OUT.exists() or OUT.read_text() != fresh:
            print(f"{OUT} is out of date; run python site/build_results.py")
            return 1
        return 0
    OUT.write_text(fresh)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
