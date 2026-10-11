"""Tests for the ``phytotron`` command's wiring (no simulator needed)."""

from pathlib import Path

import pytest

from phytotron import config as config_module
from phytotron.cli import COMMANDS, build_parser, main
from phytotron.commands.run_benchmark import registered_arms
from phytotron.config import default_config_path


@pytest.mark.filterwarnings("ignore:pin_threads")  # numpy is already loaded under pytest
@pytest.mark.parametrize("command", COMMANDS)
def test_every_subcommand_has_help(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--help"])
    assert exit_info.value.code == 0
    assert "--help" in capsys.readouterr().out


def test_run_requires_a_run_dir(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["run"])
    assert exit_info.value.code == 2
    assert "--run-dir" in capsys.readouterr().err


def test_commands_share_the_config_default() -> None:
    parser = build_parser()
    assert parser.parse_args(["run", "--run-dir", "x"]).config == default_config_path()
    assert parser.parse_args(["quickstart"]).config == default_config_path()
    custom = parser.parse_args(["quickstart", "--config", "other.yaml"]).config
    assert custom == Path("other.yaml")


def test_default_config_falls_back_to_the_packaged_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "_REPO_CONFIG", Path("/nonexistent/benchmark.yaml"))
    monkeypatch.chdir("/")
    assert default_config_path({}) == config_module._PACKAGED_CONFIG


def test_default_config_path_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert default_config_path({"PHYTOTRON_CONFIG": "/x/b.yaml"}) == Path("/x/b.yaml")
    monkeypatch.chdir(tmp_path)
    repo_default = default_config_path({})
    assert repo_default.name == "benchmark.yaml" and repo_default.is_file()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "benchmark.yaml").write_text("")
    assert default_config_path({}) == tmp_path / "configs" / "benchmark.yaml"


def test_run_parses_user_controller_options() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--run-dir",
            "x",
            "--controller",
            "still=user_ctrl:Still",
            "--controller",
            "mine=user_ctrl:make",
            "--model-based",
            "mine",
        ]
    )
    assert args.arms is None  # default: reference arms plus every --controller
    registered = registered_arms(args)
    assert set(registered) == {"still", "mine"}
    assert registered["mine"].model_based and not registered["still"].model_based
    bad = build_parser().parse_args(["run", "--run-dir", "x", "--controller", "incumbent=m:C"])
    with pytest.raises(SystemExit, match="built-in"):
        registered_arms(bad)
