"""The safety net for changes on the scoring path.

Two comparisons. The golden run: seven stored synthetic records (five seasons, two failures)
(``tests/fixtures/run-small``, see ``regenerate.py`` there) must summarise to the stored
``summary.json``, key for key and number for number. The run of 10 October 2026: its 108 season
records of run 2, when present locally, rescored under the shipped config, must reproduce
``results/2026-10-10-run-2/summary-protocol-2.json``, except for keys added since it was
stored (``KEYS_ADDED_SINCE_RUN_2``). Any difference is a change to what
the benchmark reports and must be intended, named in the commit, and the fixture
regenerated on purpose.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from phytotron.config import load_config
from phytotron.controllers.registry import specs_from_record
from phytotron.runner import load_records, rescore_records, summarize

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config(ROOT / "configs" / "benchmark.yaml")
GOLDEN = ROOT / "tests" / "fixtures" / "run-small"
RUN_2 = ROOT / "runs" / "full-2"  # the per-season records are not in the repository
STORED_RUN_2 = ROOT / "results" / "2026-10-10-run-2" / "summary-protocol-2.json"

RULE_KINDS = ("light_mean", "rolling_mean", "dose", "peak", "daily_hours", "longest_run")
# Summary keys that did not exist when the summary of the run of 10 October 2026 was
# stored. Each entry names the change that added it; the stored file is not rewritten.
KEYS_ADDED_SINCE_RUN_2 = frozenset(
    {
        "seasons_not_started",  # per arm: seasons the harness could not start (0.1.0)
        "counted",  # per failure: whether it counts against the gate (0.1.0)
    }
)


def _differences(
    expected: Any, actual: Any, path: str = "", added: frozenset[str] = frozenset()
) -> list[str]:
    """Every path where two JSON values differ (type, key set, length or value).

    ``added`` names keys that may be present in ``actual`` and absent from ``expected``:
    keys the summary gained after ``expected`` was written.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _dict_differences(expected, actual, path, added)
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: {len(expected)} items expected, {len(actual)} found"]
        pairs = zip(expected, actual, strict=True)
        return [
            d for i, (e, a) in enumerate(pairs) for d in _differences(e, a, f"{path}[{i}]", added)
        ]
    same = (
        math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-15)
        if isinstance(expected, float) and isinstance(actual, float)
        else type(expected) is type(actual) and expected == actual
    )
    return [] if same else [f"{path}: {expected!r} expected, {actual!r} found"]


def _dict_differences(
    expected: dict[str, Any], actual: dict[str, Any], path: str, added: frozenset[str]
) -> list[str]:
    out = []
    for key in sorted(set(expected) | set(actual)):
        if key not in actual:
            out.append(f"{path}/{key}: missing")
        elif key not in expected:
            if key not in added:
                out.append(f"{path}/{key}: unexpected")
        else:
            out.extend(_differences(expected[key], actual[key], f"{path}/{key}", added))
    return out


def _as_json(value: Any) -> Any:
    """``value`` as it comes back from a JSON file (tuples become lists)."""
    return json.loads(json.dumps(value))


def test_golden_run_summarises_to_the_stored_summary() -> None:
    records = load_records(GOLDEN)
    launch = json.loads((GOLDEN / "run_record.json").read_text())
    assert list(records) == launch["grid"]
    summary = _as_json(summarize(records, CFG, launch))
    stored = json.loads((GOLDEN / "summary.json").read_text())
    assert _differences(stored, summary) == []


def test_golden_run_exercises_every_rule_kind_and_both_record_paths() -> None:
    # The fixture must stay a real test of the crop gate: each rule kind broken by at
    # least one season, one season without traces, one failure with an attempt history.
    summary = json.loads((GOLDEN / "summary.json").read_text())
    rules = {
        name: spec for cls in CFG.gate.soft_classes for name, spec in CFG.gate.rules(cls).items()
    }
    broken = {
        rules[name]["kind"]
        for row in summary["seasons"]
        if row["broken_rules"]
        for names in row["broken_rules"].values()
        for name in names
    }
    assert broken == set(RULE_KINDS)
    assert any(row["broken_rules"] is None for row in summary["seasons"])
    counted = {f["stage"]: f["counted"] for f in summary["record"]["failed_seasons"]}
    assert counted == {"season": True, "setup": False}
    records = load_records(GOLDEN)
    attempts = [len(r["attempts"]) for r in records.values() if "error" in r]
    assert sorted(attempts) == [0, 1]


@pytest.mark.skipif(
    not (RUN_2 / "seasons").is_dir(), reason="run 2 season records are kept outside the repository"
)
def test_rescore_of_run_2_reproduces_the_stored_protocol_2_summary() -> None:
    launch = json.loads((RUN_2 / "run_record.json").read_text())
    records = rescore_records(load_records(RUN_2), CFG, specs_from_record(launch))
    summary = _as_json(summarize(records, CFG, launch))
    stored = json.loads(STORED_RUN_2.read_text())
    assert _differences(stored, summary, added=KEYS_ADDED_SINCE_RUN_2) == []
    # the keys added since the run was stored carry the no-failure values for that run
    assert all(a["seasons_not_started"] == 0 for a in summary["arms"].values())
    assert summary["record"]["failed_seasons"] == []
