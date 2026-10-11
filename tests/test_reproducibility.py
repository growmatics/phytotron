"""Thread pinning: variables set, late call warned, settings reported."""

import pytest

from phytotron.reproducibility import THREAD_VARS, pin_threads, thread_settings, threads_pinned


def test_pin_threads_sets_every_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    assert not threads_pinned()
    with pytest.warns(RuntimeWarning):  # numpy is already loaded inside pytest
        pin_threads()
    assert thread_settings() == dict.fromkeys(THREAD_VARS, "1")
    assert threads_pinned()


def test_pin_threads_accepts_other_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.warns(RuntimeWarning):
        pin_threads(4)
    assert set(thread_settings().values()) == {"4"}
    assert not threads_pinned()


def test_unset_variables_report_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in THREAD_VARS:
        monkeypatch.delenv(var, raising=False)
    assert thread_settings() == dict.fromkeys(THREAD_VARS)
