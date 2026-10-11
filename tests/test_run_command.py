"""End-to-end tests of ``phytotron run`` without a simulator.

A fake season runner stands in for ``run_task_safe`` and a serial executor for the
process pool, so the command's own logic is exercised: the grid and run record it
writes, resume, ``--retry-failed`` with the attempt history, the three refusals (changed
config, changed user-controller code, different grid) and ``--rescore``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phytotron.commands import run_benchmark
from phytotron.commands.run_benchmark import build_parser, main, run
from phytotron.config import load_config
from phytotron.envs.glg2 import SeasonResult
from phytotron.grid import Task
from phytotron.records import FailureRecord, Record, season_record
from phytotron.season import Paths

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "benchmark.yaml"
GRID = ["--arms", "incumbent", "physics_mpc", "--years", "2025", "--mismatch", "0", "0.1"]
SEEDS = ["--seeds", "0", "1", "--season-days", "1"]
STEPS = 4


class _SerialExecutor:
    """The process pool's shape (``submit`` and a context manager), run in order here."""

    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> _SerialExecutor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future[Any]:
        future: Future[Any] = Future()
        future.set_result(fn(*args))
        return future


class _FakeSeasons:
    """Stands in for ``run_task_safe``: a calm season; seed 1 fails on its first attempt.

    With ``fail_arm`` set, every season of that arm fails, so the arm has no number.
    """

    def __init__(self, fail_arm: str | None = None) -> None:
        self.calls: list[str] = []
        self.failed_once: set[str] = set()
        self.fail_arm = fail_arm

    def __call__(self, task: Task, paths: Paths) -> Record:
        self.calls.append(task.key)
        fails = task.arm == self.fail_arm or task.seed == 1
        if fails and task.key not in self.failed_once:
            self.failed_once.add(task.key)
            return FailureRecord(
                task=task,
                label=task.label,
                error="RuntimeError: boom",
                stage="season",
                failed_at="2026-10-11T00:00:00+00:00",
            )
        zeros = np.zeros(STEPS)
        result = SeasonResult(
            net_economic_value=1.0 - task.mismatch,
            exceedances={"humidity": zeros, "temperature": zeros, "co2": zeros},
            online_cost_seconds=0.0,
            solve_success=None,
            steps=STEPS,
        )
        return season_record(task, result)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """A working directory with a copy of the config and a weather file for 2025."""
    config = tmp_path / "configs" / "benchmark.yaml"
    config.parent.mkdir()
    config.write_text(CONFIG.read_text())
    location = load_config(CONFIG).generalisation.location
    weather = tmp_path / "weather" / location
    weather.mkdir(parents=True)
    (weather / "2025.csv").write_text("fake weather\n")
    monkeypatch.delenv("GLG2_CONFIG_DIR", raising=False)  # the shipped scenario files
    monkeypatch.setenv("GLG2_WEATHER_DIR", str(tmp_path / "weather"))
    monkeypatch.chdir(tmp_path)
    return {"config": config, "run_dir": tmp_path / "runs" / "e2e"}


def _args(ws: dict[str, Path], *extra: str, config: Path | None = None) -> argparse.Namespace:
    config = ws["config"] if config is None else config
    argv = ["--run-dir", str(ws["run_dir"]), "--config", str(config), *GRID, *SEEDS, *extra]
    return build_parser().parse_args(argv)


def _run(args: argparse.Namespace, fake: _FakeSeasons) -> int:
    return run(args, season_runner=fake, executor=_SerialExecutor)


def test_fresh_run_writes_grid_record_seasons_and_summary(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _FakeSeasons()
    code = _run(_args(workspace), fake)
    run_dir = workspace["run_dir"]
    grid = json.loads((run_dir / "grid.json").read_text())
    # incumbent and MPC at mismatch 0 run one seed (seed-invariant); the MPC at 0.1 runs two
    assert grid == [
        "incumbent__2025__m0.000__s0",
        "physics_mpc__2025__m0.000__s0",
        "physics_mpc__2025__m0.100__s0",
        "physics_mpc__2025__m0.100__s1",
    ]
    assert fake.calls and sorted(fake.calls) == grid
    record = json.loads((run_dir / "run_record.json").read_text())
    assert record["season_days"] == 1 and record["smoke"] is True
    assert record["custom_arms"] == {} and record["grid"]["seasons"] == 4
    location = load_config(CONFIG).generalisation.location
    assert set(record["inputs_sha256"]) == {
        "glg2/envs/TomatoEnv.yml",
        "glg2/agents/rule_based.yml",
        f"weather/{location}/2025.csv",
    }
    assert sorted(p.stem for p in (run_dir / "seasons").glob("*.json")) == grid
    summary = json.loads((run_dir / "summary.json").read_text())
    mpc = summary["arms"]["physics_mpc"]
    assert (mpc["n_seasons"], mpc["n_failed"], mpc["seasons_not_started"]) == (2, 1, 0)
    assert code == 1  # a failed season makes the command exit 1
    out = capsys.readouterr().out
    assert "4 seasons in the grid, 4 to run" in out
    assert "FAILED at season: RuntimeError: boom" in out
    assert "SMOKE RUN" in out and "physics_mpc: 2 seasons, 1 failed" in out


def test_resume_runs_nothing_and_keeps_the_failure(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    fake = _FakeSeasons()
    _run(_args(workspace), fake)
    first = (workspace["run_dir"] / "summary.json").read_text()
    calls = len(fake.calls)
    assert _run(_args(workspace), fake) == 1
    assert len(fake.calls) == calls
    assert "4 seasons in the grid, 0 to run" in capsys.readouterr().out
    assert (workspace["run_dir"] / "summary.json").read_text() == first


def test_retry_failed_reruns_the_failure_and_keeps_its_attempt(
    workspace: dict[str, Path],
) -> None:
    fake = _FakeSeasons()
    _run(_args(workspace), fake)
    assert _run(_args(workspace, "--retry-failed"), fake) == 0
    assert fake.calls.count("physics_mpc__2025__m0.100__s1") == 2
    record = json.loads(
        (workspace["run_dir"] / "seasons" / "physics_mpc__2025__m0.100__s1.json").read_text()
    )
    assert "error" not in record
    assert [a["stage"] for a in record["attempts"]] == ["season"]
    summary = json.loads((workspace["run_dir"] / "summary.json").read_text())
    assert summary["record"]["n_failed_seasons"] == 0


def test_resume_refuses_a_changed_config(workspace: dict[str, Path]) -> None:
    _run(_args(workspace), _FakeSeasons())
    edited = workspace["config"].with_name("edited.yaml")
    edited.write_text(workspace["config"].read_text() + "# a comment changes the hash\n")
    with pytest.raises(SystemExit, match="different config_sha256"):
        _run(_args(workspace, config=edited), _FakeSeasons())


def test_resume_refuses_a_different_grid(workspace: dict[str, Path]) -> None:
    _run(_args(workspace), _FakeSeasons())
    args = _args(workspace)
    args.seeds = [0]
    with pytest.raises(SystemExit, match="holds a different grid"):
        _run(args, _FakeSeasons())


def test_resume_refuses_changed_user_controller_code(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    module = tmp_path / "ctrl_e2e.py"
    module.write_text(
        "import numpy as np\n\n\nclass Thermostat:\n"
        "    def reset(self, seed=0):\n        pass\n\n"
        "    def act(self, ctx):\n        return np.zeros(6)\n"
    )
    spec = ["--controller", "thermostat=ctrl_e2e:Thermostat"]
    args = _args(workspace, *spec)
    args.arms = ["incumbent", "thermostat"]
    assert _run(args, _FakeSeasons()) == 0
    record = json.loads((workspace["run_dir"] / "run_record.json").read_text())
    assert record["custom_arms"]["thermostat"]["file"] == "ctrl_e2e.py"
    module.write_text(module.read_text() + "# changed\n")
    again = _args(workspace, *spec)
    again.arms = ["incumbent", "thermostat"]
    with pytest.raises(SystemExit, match="different user controllers"):
        _run(again, _FakeSeasons())


def test_rescore_writes_a_summary_under_the_current_rules(workspace: dict[str, Path]) -> None:
    _run(_args(workspace), _FakeSeasons())
    assert _run(_args(workspace, "--rescore"), _FakeSeasons()) == 1
    rescored = list(workspace["run_dir"].glob("summary-rescored-*.json"))
    assert len(rescored) == 1
    original = json.loads((workspace["run_dir"] / "summary.json").read_text())
    assert json.loads(rescored[0].read_text())["arms"] == original["arms"]


def test_main_runs_the_command(workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeSeasons()
    monkeypatch.setattr(run_benchmark, "run_task_safe", fake)
    monkeypatch.setattr(run_benchmark, "ProcessPoolExecutor", _SerialExecutor)
    argv = ["--run-dir", str(workspace["run_dir"]), "--config", str(workspace["config"])]
    assert main([*argv, *GRID, "--seeds", "0", "--season-days", "1"]) == 0
    assert len(fake.calls) == 3


def test_an_arm_without_a_completed_season_is_written_and_printed_without_nan(
    workspace: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(_args(workspace), _FakeSeasons(fail_arm="physics_mpc")) == 1
    text = (workspace["run_dir"] / "summary.json").read_text()
    assert "NaN" not in text
    mpc = json.loads(text)["arms"]["physics_mpc"]
    assert (mpc["n_seasons"], mpc["n_failed"]) == (0, 3)
    assert mpc["net_value_mean"] is None and mpc["verdict"]["net_value_mean"] is None
    assert mpc["online_cost_seconds"] is None
    pareto = json.loads(text)["pareto"]
    assert pareto["ungated"]["arms"] == ["incumbent"]  # no point to rank the MPC on
    assert pareto["ungated"]["non_dominated"] == ["incumbent"]
    assert "net   n/a +/-   n/a EUR/m2" in capsys.readouterr().out
