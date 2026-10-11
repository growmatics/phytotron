"""Tests that the docs match the code: the crop-rule table matches the config, every file
path the code and the docs mention exists, and every relative link between docs resolves."""

import re
from pathlib import Path

import pytest

from phytotron.config import load_config

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config(ROOT / "configs" / "benchmark.yaml")
KPIS = (ROOT / "docs" / "kpis.md").read_text()
RULES = [(c, name, spec) for c in CFG.gate.soft_classes for name, spec in CFG.gate.rules(c).items()]


def _row(name: str) -> str:
    rows = [line for line in KPIS.splitlines() if line.startswith("|") and f"`{name}`" in line]
    assert len(rows) == 1, f"docs/kpis.md needs exactly one table row for `{name}`"
    return rows[0]


@pytest.mark.parametrize(("cls", "name", "spec"), RULES, ids=[r[1] for r in RULES])
def test_every_rule_number_is_in_its_doc_row(cls: str, name: str, spec: dict) -> None:
    row = _row(name)
    assert row.split("|")[1].strip().lower() == cls
    numbers = set(re.findall(r"\d+(?:\.\d+)?", row))
    for key, value in spec.items():
        if key != "kind":
            assert f"{float(value):g}" in numbers, f"{name}.{key} = {value} missing from the doc"


def test_doc_table_has_no_rule_the_config_lacks() -> None:
    names = set(re.findall(r"^\| \w+ \| `(\w+)` \|", KPIS, flags=re.MULTILINE))
    assert names == {name for _c, name, _s in RULES}


SOURCES = sorted(
    [
        *ROOT.glob("phytotron/**/*.py"),
        *ROOT.glob("examples/*.py"),
        *ROOT.glob("site/*.py"),
        *ROOT.glob("docs/*.md"),
        *ROOT.glob("site/content/*.md"),
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
        ROOT / "configs" / "benchmark.yaml",
    ]
)
# A repository path with an extension, as docstrings and docs write them.
PATH_MENTION = re.compile(
    r"\b((?:docs|examples|site|configs|results|tests|phytotron)/[\w./-]+\.(?:md|py|yaml|json|svg|txt|cff))\b"
)
DOC_LINK = re.compile(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)")
DOCS = sorted([*ROOT.glob("docs/*.md"), ROOT / "README.md"])


@pytest.mark.parametrize("source", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_file_path_mentioned_exists(source: Path) -> None:
    # A docstring or a doc that names a file that is not there sends the reader nowhere.
    mentioned = set(PATH_MENTION.findall(source.read_text()))
    missing = sorted(m for m in mentioned if not (ROOT / m).exists())
    assert missing == [], f"{source.relative_to(ROOT)} names files that do not exist: {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_every_relative_doc_link_resolves(doc: Path) -> None:
    targets = [t for t in DOC_LINK.findall(doc.read_text()) if "://" not in t]
    broken = sorted(t for t in targets if not (doc.parent / t).exists())
    assert broken == [], f"{doc.relative_to(ROOT)} links to files that do not exist: {broken}"
