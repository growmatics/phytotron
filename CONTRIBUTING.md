# Contributing

Thank you for your interest in Phytotron.

Open an issue or a pull request on this repository. Development happens in a separate working repository, so an accepted pull request is applied there and appears here in the next release, with you credited in the changelog.

Set up a development environment once:

```
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Before you open a pull request, run (from the venv):

```
ruff check .
ruff format --check .
mypy phytotron site
pytest
```

When you report results from Phytotron, say they are simulated, and do not call them a digital twin of a named greenhouse.

To report a security problem, do not open a public issue; see [SECURITY.md](SECURITY.md).

By contributing you agree that your work is licensed under AGPL-3.0-or-later.
