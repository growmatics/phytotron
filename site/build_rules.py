"""Write ``site/content/rules.json``: the crop rules as the website shows them.

Numbers come from ``configs/benchmark.yaml`` only; the wording comes from
``site/content/rule_text.yaml``, whose sentences are templates filled with those numbers.
The website renders ``rules.json`` and never types a number itself.

    python site/build_rules.py          # rewrite rules.json
    python site/build_rules.py --check  # exit 1 if rules.json is out of date
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import figures
import yaml

from phytotron.config import load_config
from phytotron.metrics.gate import clopper_pearson_upper

# Season counts shown in the tolerance table: the minimum, the MPC verdict, and the first
# counts that tolerate one, two and three failing seasons.
TOLERANCE_SEASONS = (60, 93, 100, 124, 153)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "benchmark.yaml"
TEXT = ROOT / "site" / "content" / "rule_text.yaml"
OUT = ROOT / "site" / "content" / "rules.json"


def _number(value: Any) -> str:
    return f"{float(value):g}"


def step_seconds(cfg: Any) -> int:
    """The control step of the benchmark, from the config's episode block."""
    return int(cfg.episode.timestep_seconds)


def peak_limits(cfg: Any) -> tuple[float, float]:
    """``(low, high)`` of the absolute-limit rule (kind ``peak``) among the gated classes."""
    for cls in cfg.gate.soft_classes:
        for spec in cfg.gate.rules(cls).values():
            if spec["kind"] == "peak":
                return float(spec["low"]), float(spec["high"])
    raise ValueError("no rule of kind 'peak' in the gated classes")


def _allowed(n: int, epsilon: float, delta: float) -> int | None:
    """Most failing seasons out of ``n`` that still pass (None if even 0 fails)."""
    k = 0
    while k <= n and clopper_pearson_upper(k, n, delta) <= epsilon:
        k += 1
    return k - 1 if k > 0 else None


def build() -> dict[str, Any]:
    """The rules document, built from the config and the rule text."""
    cfg = load_config(CONFIG)
    text = yaml.safe_load(TEXT.read_text())
    sources = text["sources"]
    classes = []
    for cls in cfg.gate.soft_classes:
        rules = []
        for key, spec in cfg.gate.rules(cls).items():
            if key not in text["rules"]:
                raise KeyError(f"rule {key!r} has no text in {TEXT.name}")
            words = text["rules"][key]
            params = {k: v for k, v in spec.items() if k != "kind"}
            condition = " ".join(words["sentence"].split()).format(
                **{k: _number(v) for k, v in params.items()}
            )
            refs = [{"key": ref, **sources[ref]} for ref in words["sources"]]
            rules.append(
                {
                    "key": key,
                    "label": words["label"],
                    "harm": words["harm"],
                    "fails_when": condition,
                    "kind": spec["kind"],
                    "params": params,
                    "sources": refs,
                }
            )
        classes.append({"class": cls, "rules": rules})
    reported = [c.name for c in cfg.gate.classes if not c.gated]
    return {
        "generated_from": {
            "config": "configs/benchmark.yaml",
            "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            "rules_status": cfg.gate.rules_status,
            "protocol_version": cfg.gate.protocol_version,
        },
        "gate": {
            "trial_unit": "season",
            "confidence": 1.0 - cfg.gate.delta,
            "max_failing_share": cfg.gate.epsilon,
            "min_seasons": cfg.gate.n_seasons_min,
            "season_fails_class_if": "any rule of the class breaks",
            "crashed_season_counts_as": "failing",
            "season_not_started_counts_as": "not started, listed",
            "failing_seasons_allowed": [
                {"seasons": n, "allowed": _allowed(n, cfg.gate.epsilon, cfg.gate.delta)}
                for n in TOLERANCE_SEASONS
            ],
        },
        "gated_classes": classes,
        "reported_classes": reported,
        "sources": [{"key": k, **sources[k]} for k in sorted(sources)],
    }


def render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if rules.json is stale")
    args = parser.parse_args()
    doc = build()
    outputs = {OUT: render(doc)}
    figures_dir = figures.FIGURES_DIR
    outputs.update({figures_dir / name: svg for name, svg in figures.render_figures(doc).items()})
    if args.check:
        stale = [p for p, fresh in outputs.items() if not p.exists() or p.read_text() != fresh]
        for path in stale:
            print(f"{path} is out of date; run python site/build_rules.py")
        return 1 if stale else 0
    figures_dir.mkdir(exist_ok=True)
    for path, fresh in outputs.items():
        path.write_text(fresh)
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
