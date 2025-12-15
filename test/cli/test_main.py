"""Tests for slm_lab.cli.main module."""

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from slm_lab.cli import app
from slm_lab.cli.main import set_variables, get_spec, stop_ray_processes, _lazy_imports


runner = CliRunner()


@pytest.fixture(scope="module", autouse=False)
def lazy_imports():
    """Load lazy imports needed by get_spec and other functions."""
    _lazy_imports()


class TestSetVariables:
    """Tests for set_variables helper function."""

    def test_no_sets_returns_spec_unchanged(self):
        spec = {"agent": {"algorithm": {"lr": 0.001}}}
        result = set_variables(spec, None)
        assert result == spec

    def test_empty_sets_returns_spec_unchanged(self):
        spec = {"agent": {"algorithm": {"lr": 0.001}}}
        result = set_variables(spec, [])
        assert result == spec

    def test_single_variable_substitution(self):
        spec = {"env": {"name": "${env}"}}
        result = set_variables(spec, ["env=CartPole-v1"])
        assert result["env"]["name"] == "CartPole-v1"

    def test_multiple_variable_substitution(self):
        spec = {"env": {"name": "${env}"}, "agent": {"algorithm": {"lr": "${lr}"}}}
        result = set_variables(spec, ["env=CartPole-v1", "lr=0.0003"])
        assert result["env"]["name"] == "CartPole-v1"
        assert result["agent"]["algorithm"]["lr"] == "0.0003"

    def test_variable_with_equals_in_value(self):
        spec = {"meta": {"note": "${note}"}}
        result = set_variables(spec, ["note=key=value"])
        assert result["meta"]["note"] == "key=value"

    def test_unmatched_variable_unchanged(self):
        spec = {"env": {"name": "${env}"}, "other": "${other}"}
        result = set_variables(spec, ["env=CartPole-v1"])
        assert result["env"]["name"] == "CartPole-v1"
        assert result["other"] == "${other}"

    def test_nested_variable_substitution(self):
        spec = {"a": {"b": {"c": {"d": "${deep}"}}}}
        result = set_variables(spec, ["deep=value"])
        assert result["a"]["b"]["c"]["d"] == "value"

    def test_env_appends_shortname_with_ale_prefix(self):
        spec = {"name": "ppo_atari", "env": {"name": "${env}"}}
        result = set_variables(spec, ["env=ALE/Pong-v5"])
        assert result["name"] == "ppo_atari_pong"
        assert result["env"]["name"] == "ALE/Pong-v5"

    def test_env_appends_shortname_without_prefix(self):
        spec = {"name": "ppo_mujoco", "env": {"name": "${env}"}}
        result = set_variables(spec, ["env=HalfCheetah-v5"])
        assert result["name"] == "ppo_mujoco_halfcheetah"

    def test_non_env_var_does_not_append_shortname(self):
        spec = {"name": "test", "lr": "${lr}"}
        result = set_variables(spec, ["lr=0.001"])
        assert result["name"] == "test"

    def test_env_with_multiple_vars_appends_shortname(self):
        spec = {"name": "ppo_atari", "env": {"name": "${env}"}, "lr": "${lr}"}
        result = set_variables(spec, ["env=ALE/Qbert-v5", "lr=0.0003"])
        assert result["name"] == "ppo_atari_qbert"
        assert result["lr"] == "0.0003"


@pytest.mark.usefixtures("lazy_imports")
class TestGetSpec:
    """Tests for get_spec function."""

    @patch("slm_lab.cli.main.spec_util")
    def test_train_mode_new_trial(self, mock_spec_util):
        mock_spec_util.get.return_value = {"meta": {}, "agent": {}}
        result = get_spec("spec.json", "test_spec", "train", None, None)
        mock_spec_util.get.assert_called_once_with("spec.json", "test_spec")
        assert result == {"meta": {}, "agent": {}}

    @patch("slm_lab.cli.main.spec_util")
    def test_dev_mode_new_trial(self, mock_spec_util):
        mock_spec_util.get.return_value = {"meta": {}, "agent": {}}
        get_spec("spec.json", "test_spec", "dev", None, None)
        mock_spec_util.get.assert_called_once_with("spec.json", "test_spec")

    @patch("slm_lab.cli.main.spec_util")
    def test_search_mode_new_trial(self, mock_spec_util):
        mock_spec_util.get.return_value = {"meta": {}, "agent": {}}
        get_spec("spec.json", "test_spec", "search", None, None)
        mock_spec_util.get.assert_called_once_with("spec.json", "test_spec")

    @patch("slm_lab.cli.main.util")
    @patch("slm_lab.cli.main.spec_util")
    def test_train_mode_resume(self, mock_spec_util, mock_util):
        mock_util.get_experiment_ts.return_value = "2024_01_01_120000"
        mock_spec_util.get.return_value = {"meta": {}, "agent": {}}
        get_spec("spec.json", "test_spec", "train", "data/test_spec_2024", None)
        mock_util.get_experiment_ts.assert_called_once_with("data/test_spec_2024")
        mock_spec_util.get.assert_called_once_with(
            "spec.json", "test_spec", "2024_01_01_120000"
        )

    @patch("slm_lab.cli.main.util")
    def test_enjoy_mode(self, mock_util):
        mock_util.read.return_value = {"meta": {"predir": "data/test"}}
        result = get_spec("spec.json", "test_spec", "enjoy", "session_spec.json", None)
        mock_util.read.assert_called_once_with("session_spec.json")
        assert result == {"meta": {"predir": "data/test"}}

    def test_enjoy_mode_requires_predir(self):
        with pytest.raises(AssertionError, match="enjoy mode must specify"):
            get_spec("spec.json", "test_spec", "enjoy", None, None)

    def test_invalid_mode_raises_error(self):
        with pytest.raises(ValueError, match="Unrecognizable lab_mode"):
            get_spec("spec.json", "test_spec", "invalid_mode", None, None)

    @patch("slm_lab.cli.main.spec_util")
    def test_with_variable_sets(self, mock_spec_util):
        mock_spec_util.get.return_value = {"env": {"name": "${env}"}}
        result = get_spec("spec.json", "test_spec", "train", None, ["env=CartPole-v1"])
        assert result["env"]["name"] == "CartPole-v1"


@pytest.mark.usefixtures("lazy_imports")
class TestStopRayProcesses:
    """Tests for stop_ray_processes function."""

    @patch("slm_lab.cli.main.subprocess.run")
    def test_calls_ray_stop(self, mock_run):
        stop_ray_processes()
        # Should call ray stop first
        assert mock_run.call_count >= 1
        first_call = mock_run.call_args_list[0]
        assert "ray" in first_call[0][0]
        assert "stop" in first_call[0][0]


class TestCliCommands:
    """Tests for CLI commands using typer.testing."""

    def test_help_command(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Modular deep reinforcement learning framework" in result.output

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run SLM-Lab experiments locally" in result.output
        assert "--render" in result.output
        assert "--log-level" in result.output
        assert "--set" in result.output
        assert "--job" in result.output
        assert "--keep" in result.output

    def test_run_remote_help(self):
        result = runner.invoke(app, ["run-remote", "--help"])
        assert result.exit_code == 0
        assert "Launch experiment on dstack" in result.output
        assert "--name" in result.output
        assert "--gpu" in result.output
        assert "--set" in result.output

    def test_pull_help(self):
        result = runner.invoke(app, ["pull", "--help"])
        assert result.exit_code == 0
        assert "Pull experiment results from HuggingFace" in result.output
        assert "--list" in result.output

    def test_push_help(self):
        result = runner.invoke(app, ["push", "--help"])
        assert result.exit_code == 0
        assert "Push local experiment to HuggingFace" in result.output

    def test_list_help(self):
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0
        assert "List experiments available on HuggingFace" in result.output

    @patch("slm_lab.cli.main.stop_ray_processes")
    def test_run_stop_ray_flag(self, mock_stop):
        result = runner.invoke(app, ["run", "--stop-ray"])
        assert result.exit_code == 0
        mock_stop.assert_called_once()

    def test_all_commands_registered(self):
        result = runner.invoke(app, ["--help"])
        assert "run " in result.output or "run" in result.output
        assert "run-remote" in result.output
        assert "pull" in result.output
        assert "push" in result.output
        assert "list" in result.output
