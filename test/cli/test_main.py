"""Tests for slm_lab.cli.main module."""

import re
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from slm_lab.cli import app
from slm_lab.cli.main import get_spec, stop_ray_processes, _lazy_imports, find_saved_spec
from slm_lab.spec.spec_util import set_variables


runner = CliRunner()

# Strip ANSI color codes for consistent test assertions
ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')
def strip_ansi(text):
    return ANSI_PATTERN.sub('', text)


@pytest.fixture(scope="module", autouse=False)
def lazy_imports():
    """Load lazy imports needed by get_spec and other functions."""
    _lazy_imports()


class TestSetVariables:
    """Tests for set_variables helper function."""

    def test_no_sets_returns_unchanged(self):
        spec_str = '{"agent": {"algorithm": {"lr": 0.001}}}'
        result, env_short = set_variables(spec_str, None)
        assert result == spec_str
        assert env_short is None

    def test_empty_sets_returns_unchanged(self):
        spec_str = '{"agent": {"algorithm": {"lr": 0.001}}}'
        result, env_short = set_variables(spec_str, [])
        assert result == spec_str
        assert env_short is None

    def test_single_variable_substitution(self):
        spec_str = '{"env": {"name": "${env}"}}'
        result, _ = set_variables(spec_str, ["env=CartPole-v1"])
        assert '"name": "CartPole-v1"' in result

    def test_numeric_variable_substitution(self):
        spec_str = '{"env": {"max_frame": "${max_frame}"}}'
        result, _ = set_variables(spec_str, ["max_frame=3e6"])
        assert '"max_frame": 3e6' in result  # unquoted number

    def test_variable_with_equals_in_value(self):
        spec_str = '{"meta": {"note": "${note}"}}'
        result, _ = set_variables(spec_str, ["note=key=value"])
        assert '"note": "key=value"' in result

    def test_unmatched_variable_unchanged(self):
        spec_str = '{"env": {"name": "${env}"}, "other": "${other}"}'
        result, _ = set_variables(spec_str, ["env=CartPole-v1"])
        assert '"name": "CartPole-v1"' in result
        assert '"other": "${other}"' in result

    def test_env_returns_shortname(self):
        spec_str = '{"env": {"name": "${env}"}}'
        _, env_short = set_variables(spec_str, ["env=ALE/Pong-v5"])
        assert env_short == "pong"

    def test_env_shortname_without_prefix(self):
        spec_str = '{"env": {"name": "${env}"}}'
        _, env_short = set_variables(spec_str, ["env=HalfCheetah-v5"])
        assert env_short == "halfcheetah"

    def test_non_env_var_no_shortname(self):
        spec_str = '{"lr": "${lr}"}'
        _, env_short = set_variables(spec_str, ["lr=0.001"])
        assert env_short is None


class TestFindSavedSpec:
    """Tests for find_saved_spec function."""

    def test_returns_latest_trial_spec(self, tmp_path):
        predir = tmp_path / "data" / "ppo_test_2024"
        predir.mkdir(parents=True)
        (predir / "ppo_test_t0_spec.json").touch()
        (predir / "ppo_test_t1_spec.json").touch()
        result = find_saved_spec(str(predir), "ppo_test")
        assert result == str(predir / "ppo_test_t1_spec.json")

    def test_returns_none_if_no_match(self, tmp_path):
        predir = tmp_path / "empty"
        predir.mkdir()
        result = find_saved_spec(str(predir), "ppo_test")
        assert result is None

    def test_returns_single_spec(self, tmp_path):
        predir = tmp_path / "data"
        predir.mkdir()
        (predir / "ppo_test_t0_spec.json").touch()
        result = find_saved_spec(str(predir), "ppo_test")
        assert result == str(predir / "ppo_test_t0_spec.json")


@pytest.mark.usefixtures("lazy_imports")
class TestGetSpec:
    """Tests for get_spec function."""

    @patch("slm_lab.cli.main.spec_util")
    def test_train_mode_new_trial(self, mock_spec_util):
        mock_spec_util.get.return_value = {"meta": {}, "agent": {}}
        result = get_spec("spec.json", "test_spec", "train", None, None)
        mock_spec_util.get.assert_called_once_with("spec.json", "test_spec", sets=None)
        assert result == {"meta": {}, "agent": {}}

    @patch("slm_lab.cli.main.spec_util")
    def test_dev_mode_new_trial(self, mock_spec_util):
        mock_spec_util.get.return_value = {"meta": {}, "agent": {}}
        get_spec("spec.json", "test_spec", "dev", None, None)
        mock_spec_util.get.assert_called_once_with("spec.json", "test_spec", sets=None)

    @patch("slm_lab.cli.main.spec_util")
    def test_search_mode_new_trial(self, mock_spec_util):
        mock_spec_util.get.return_value = {"meta": {}, "agent": {}}
        get_spec("spec.json", "test_spec", "search", None, None)
        mock_spec_util.get.assert_called_once_with("spec.json", "test_spec", sets=None)

    @patch("slm_lab.cli.main.util")
    @patch("slm_lab.cli.main.spec_util")
    def test_train_mode_resume(self, mock_spec_util, mock_util):
        mock_util.get_experiment_ts.return_value = "2024_01_01_120000"
        mock_spec_util.get.return_value = {"meta": {}, "agent": {}}
        get_spec("spec.json", "test_spec", "train", "data/test_spec_2024", None)
        mock_util.get_experiment_ts.assert_called_once_with("data/test_spec_2024")
        mock_spec_util.get.assert_called_once_with(
            "spec.json", "test_spec", "2024_01_01_120000", sets=None
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
        mock_spec_util.get.return_value = {"env": {"name": "CartPole-v1"}}
        result = get_spec("spec.json", "test_spec", "train", None, ["env=CartPole-v1"])
        mock_spec_util.get.assert_called_once_with("spec.json", "test_spec", sets=["env=CartPole-v1"])
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
        output = strip_ansi(result.output)
        assert "Run SLM-Lab experiments locally" in output
        assert "--render" in output
        assert "--log-level" in output
        assert "--set" in output
        assert "--keep" in output

    def test_run_remote_help(self):
        result = runner.invoke(app, ["run-remote", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "Launch experiment on dstack" in output
        assert "--name" in output
        assert "--gpu" in output
        assert "--set" in output

    def test_pull_help(self):
        result = runner.invoke(app, ["pull", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "Pull experiment results from HuggingFace" in output
        assert "--list" in output

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
