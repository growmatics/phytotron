"""Tests that the website content in site/ matches configs/benchmark.yaml."""

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

from phytotron.config import load_config

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "site"))

import build_results  # noqa: E402
import build_rules  # noqa: E402
import figures  # noqa: E402

CFG = load_config(ROOT / "configs" / "benchmark.yaml")
NUMBER = r"-?\d+(?:[.,]\d+)?"
MINUS = "\u2212"  # the typographic minus used in the pages
RULES = [(c, name, spec) for c in CFG.gate.soft_classes for name, spec in CFG.gate.rules(c).items()]


def test_rules_json_and_figures_are_up_to_date() -> None:
    doc = build_rules.build()
    assert build_rules.OUT.read_text() == build_rules.render(doc), "run python site/build_rules.py"
    for name, svg in figures.render_figures(doc).items():
        assert (figures.FIGURES_DIR / name).read_text() == svg, "run python site/build_rules.py"


def test_every_figure_is_referenced_with_alt_text() -> None:
    pages = "\n".join(p.read_text() for p in (ROOT / "site" / "content").glob("*.md"))
    for name in figures.FIGURES:
        match = re.search(rf"!\[([^\]]+)\]\(figures/{re.escape(name)}\)", pages)
        assert match and match.group(1).strip(), f"{name} is not referenced with alt text"


@pytest.mark.parametrize(("cls", "name", "spec"), RULES, ids=[r[1] for r in RULES])
def test_every_rule_number_is_in_its_sentence(cls: str, name: str, spec: dict) -> None:
    doc = json.loads(build_rules.OUT.read_text())
    rules = {r["key"]: r for c in doc["gated_classes"] if c["class"] == cls for r in c["rules"]}
    numbers = set(re.findall(r"\d+(?:\.\d+)?", rules[name]["fails_when"]))
    for key, value in spec.items():
        if key != "kind":
            assert f"{float(value):g}" in numbers, f"{name}.{key} = {value} missing"
    assert rules[name]["sources"], f"{name} has no source"


def test_every_rule_has_text_and_every_source_is_used() -> None:
    text = yaml.safe_load((ROOT / "site" / "content" / "rule_text.yaml").read_text())
    assert set(text["rules"]) == {name for _c, name, _s in RULES}
    cited = {ref for words in text["rules"].values() for ref in words["sources"]}
    assert cited <= set(text["sources"])


def _numbers(text: str) -> set[float]:
    return {float(n.replace(",", "")) for n in re.findall(NUMBER, text.replace(MINUS, "-"))}


def test_results_json_is_up_to_date() -> None:
    fresh = build_results.render(build_results.build())
    assert build_results.OUT.read_text() == fresh, "run python site/build_results.py"


def test_results_were_scored_under_the_published_rules() -> None:
    """The rescored block carries the current rules; the original run may be one protocol
    behind, in which case the page shows both."""
    doc = json.loads(build_results.OUT.read_text())
    assert doc["protocol"]["gated_classes"] == list(CFG.gate.soft_classes)
    current = doc["rescored"] or doc["protocol"]
    assert current.get("protocol_version", current.get("version")) == CFG.gate.protocol_version
    for cls in CFG.gate.soft_classes:
        assert current["rules"][cls] == CFG.gate.rules(cls), cls
    if doc["rescored"]:
        assert doc["protocol"]["version"] < doc["rescored"]["protocol_version"]


@pytest.mark.parametrize("page", ["results.md", "why-crop-safety-first.md"])
def test_every_number_in_results_prose_is_in_results_json(page: str) -> None:
    """A page's sentences may repeat a number from the run, never invent one."""
    doc = json.loads(build_results.OUT.read_text())
    known = _numbers(json.dumps(doc)) | _numbers(json.dumps(CFG.raw)) | {15, 34, 50, 85, 300, 1600}
    body = (ROOT / "site" / "content" / page).read_text().split("---", 2)[2]
    body = re.sub(r"\]\([^)]*\)", "]()", body)  # link targets (release assets) are not prose
    for match in re.finditer(NUMBER, body.replace(MINUS, "-")):
        context = body[max(0, match.start() - 60) : match.end() + 20]
        if "day" in context or "minute" in context or "hour" in context or "%" in context:
            continue  # durations, dates and shares are rule or calendar numbers
        if re.fullmatch(r"(19|20)\d\d", match.group()):
            continue  # a year (a source's or a season's)
        assert float(match.group()) in known, f"{page}: {match.group()!r} in: {context.strip()}"


def test_docs_results_tables_match_results_json() -> None:
    """Every numeric cell in the docs results tables is a number of the run."""
    doc = json.loads(build_results.OUT.read_text())
    known = _numbers(json.dumps(doc)) | {60.0}  # 60 = season length in days, in the setup line
    cells: set[float] = set()
    for line in (ROOT / "docs" / "results.md").read_text().splitlines():
        if (
            line.startswith("|")
            and not line.startswith("|---")
            and not line.startswith("| Controller")
        ):
            for cell in line.strip("|").split("|")[1:]:
                for n in re.findall(NUMBER, cell.replace(MINUS, "-")):
                    if "%" not in cell and "day" not in cell and "ms" not in cell:
                        cells.add(float(n.replace(",", "")))
    missing = sorted(c for c in cells if c not in known and c not in {5761.0})
    assert not missing, missing


def test_overview_questions_and_progress_have_the_shape_the_site_renders() -> None:
    """FAQ structured data needs a bold question line and an answer paragraph; progress
    entries start with an absolute date."""
    body = (ROOT / "site" / "content" / "overview.md").read_text()
    assert "faq: true" in body.split("---", 2)[1]
    questions = body.split("## Questions", 1)[1].strip().split("\n\n")
    assert len(questions) >= 4
    for entry in questions:
        lines = entry.strip().splitlines()
        assert re.fullmatch(r"\*\*[^*]+\?\*\*", lines[0]), lines[0]
        assert len(lines) >= 2 and lines[1].strip(), lines[0]
    progress = body.split("## Progress", 1)[1].split("## Roadmap", 1)[0].strip().splitlines()
    assert progress and all(
        re.match(r"- \*\*\d{1,2} [A-Z][a-z]+ \d{4}\.\*\*", line) for line in progress
    ), progress


def test_pages_keep_the_wording_rules() -> None:
    for page in (ROOT / "site" / "content").glob("*.md"):
        body = page.read_text().lower()
        assert "simulated" in body, f"{page.name} never says simulated"
        # The phrase may appear only in the sentence that rejects it.
        for hit in re.finditer(r"digital twin", body):
            context = body[max(0, hit.start() - 80) : hit.end() + 80]
            assert "not" in context or "no." in context or "never" in context, page.name


FIXTURE_SEASONS = ROOT / "tests" / "fixtures" / "run-small" / "seasons"


def test_trace_facts_take_the_step_and_the_limits_from_the_config(tmp_path: Path) -> None:
    # The golden fixture has one MPC season at 8 C and one at 35 C, and a calm incumbent.
    facts = build_results.trace_facts(FIXTURE_SEASONS, CFG)
    assert build_rules.peak_limits(CFG) == (10.0, 34.0)
    assert facts["mpc_seasons_below_limit"] == 1
    assert facts["mpc_seasons_above_limit"] == 1
    assert facts["incumbent_dips_per_season_min_max"] == [0, 0]
    assert facts["incumbent_dip_days_min_max"] is None
    # Move the lower limit below 8 C in a copy of the config: the count follows the config.
    raw = yaml.safe_load((ROOT / "configs" / "benchmark.yaml").read_text())
    raw["gate"]["classes"]["temperature"]["rules"]["peak"]["low"] = 7.0
    raw["gate"]["classes"]["temperature"]["rules"]["dose"]["limit_low"] = 7.0
    moved = tmp_path / "benchmark.yaml"
    moved.write_text(yaml.safe_dump(raw))
    assert (
        build_results.trace_facts(FIXTURE_SEASONS, load_config(moved))["mpc_seasons_below_limit"]
        == 0
    )
