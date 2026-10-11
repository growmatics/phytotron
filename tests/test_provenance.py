"""Tests for the run record: hashes, versions, arm settings."""

import json
from pathlib import Path

from phytotron.config import load_config
from phytotron.controllers.registry import ArmSpec
from phytotron.provenance import arm_settings, file_sha256, input_hashes, package_versions
from phytotron.scenarios import glg2_config_dir
from phytotron.season import Paths

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "benchmark.yaml"
CFG = load_config(CONFIG)


def test_run_provenance(tmp_path: Path) -> None:
    envs, agents, weather = tmp_path / "envs", tmp_path / "agents", tmp_path / "weather"
    for folder in (envs, agents, weather / CFG.generalisation.location):
        folder.mkdir(parents=True)
    (envs / "TomatoEnv.yml").write_text("a: 1\n")
    (agents / "rule_based.yml").write_text("TomatoEnv: {temp_setpoint_day: 19.5}\n")
    (weather / CFG.generalisation.location / "2025.csv").write_text("t\n0\n")
    paths = Paths("cfg.yaml", str(envs), str(weather))
    hashes = input_hashes(CFG, paths, [2025, 2025])
    assert sorted(hashes) == [
        "glg2/agents/rule_based.yml",
        "glg2/envs/TomatoEnv.yml",
        f"weather/{CFG.generalisation.location}/2025.csv",
    ]
    assert hashes["glg2/envs/TomatoEnv.yml"] == file_sha256(envs / "TomatoEnv.yml")
    settings = arm_settings(CFG, paths)
    assert settings["incumbent"] == {"temp_setpoint_day": 19.5}
    assert settings["physics_mpc"]["horizon"] == CFG.mpc.horizon_steps
    json.dumps(settings)  # plain data
    versions = package_versions()
    assert {"python", "numpy", "gl-gym"} <= set(versions)


def test_arm_settings_include_user_controllers(tmp_path: Path) -> None:
    paths = Paths("configs/benchmark.yaml", str(glg2_config_dir()), str(tmp_path))
    settings = arm_settings(CFG, paths, {"still": ArmSpec("still", "user_ctrl:Still")})
    assert settings["still"] == {
        "name": "still",
        "target": "user_ctrl:Still",
        "model_based": False,
        "seed_sensitive": False,
    }
    assert "incumbent" in settings and "physics_mpc" in settings
