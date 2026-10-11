"""Phytotron: an open virtual phytotron for benchmarking greenhouse climate control.

Public API is intentionally small: the config loader, the Controller interface,
and the pure metric functions. Constants live in ``configs/benchmark.yaml`` (the
single source of truth); nothing here hardcodes a benchmark constant.

Terms used throughout the code ("arm 1" to "arm 4", "inverse crime") are defined in the
glossary in ``docs/README.md``.
"""

__version__ = "0.2.0"
