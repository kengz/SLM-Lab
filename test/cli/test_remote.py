"""Tests for slm_lab.cli.remote module."""

import re
from unittest.mock import patch

from typer.testing import CliRunner

from slm_lab.cli import app


runner = CliRunner()

# Strip ANSI color codes for consistent test assertions
ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')
def strip_ansi(text):
    return ANSI_PATTERN.sub('', text)


class TestRunRemote:
    """Tests for run_remote function via CLI runner."""

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_default_cpu_train_config(self, mock_run):
        """Test CPU train config is used by default for train mode."""
        runner.invoke(app, ["run-remote", "spec.json", "test_spec", "train"])

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        env = call_args[1]["env"]

        assert "-f" in cmd
        assert ".dstack/run-cpu-train.yml" in cmd
        assert env["SPEC_FILE"] == "spec.json"
        assert env["SPEC_NAME"] == "test_spec"
        assert env["LAB_MODE"] == "train"

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_cpu_search_config(self, mock_run):
        """Test CPU search config for search mode (default)."""
        runner.invoke(app, ["run-remote", "spec.json", "test_spec", "search"])

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert ".dstack/run-cpu-search.yml" in cmd

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_gpu_train_config(self, mock_run):
        """Test GPU train config when --gpu specified."""
        runner.invoke(app, ["run-remote", "--gpu", "spec.json", "test_spec", "train"])

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert ".dstack/run-gpu-train.yml" in cmd

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_gpu_search_config(self, mock_run):
        """Test GPU search config when --gpu specified."""
        runner.invoke(app, ["run-remote", "--gpu", "spec.json", "test_spec", "search"])

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert ".dstack/run-gpu-search.yml" in cmd

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_dev_mode_uses_train_config(self, mock_run):
        """Test that dev mode uses train config (lighter resources)."""
        runner.invoke(app, ["run-remote", "spec.json", "test_spec", "dev"])

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        # dev mode should use train config (CPU by default)
        assert ".dstack/run-cpu-train.yml" in cmd
        env = call_args[1]["env"]
        assert env["LAB_MODE"] == "dev"

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_custom_run_name(self, mock_run):
        """Test custom run name is passed correctly."""
        runner.invoke(app, ["run-remote", "spec.json", "test_spec", "train", "-n", "my-custom-run"])

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "--name" in cmd
        name_idx = cmd.index("--name")
        assert cmd[name_idx + 1] == "my-custom-run"

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_default_run_name_from_spec(self, mock_run):
        """Test default run name is derived from spec_name with underscores replaced."""
        runner.invoke(app, ["run-remote", "spec.json", "ppo_cartpole", "train"])

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        name_idx = cmd.index("--name")
        assert cmd[name_idx + 1] == "ppo-cartpole"

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_spec_vars_passed_correctly(self, mock_run):
        """Test that --set variables are passed as SPEC_VARS env."""
        runner.invoke(app, [
            "run-remote", "spec.json", "test_spec", "train",
            "-s", "env=CartPole-v1", "-s", "lr=0.001"
        ])

        call_args = mock_run.call_args
        env = call_args[1]["env"]
        assert "-s env=CartPole-v1" in env["SPEC_VARS"]
        assert "-s lr=0.001" in env["SPEC_VARS"]

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_empty_spec_vars(self, mock_run):
        """Test empty SPEC_VARS when no sets provided."""
        runner.invoke(app, ["run-remote", "spec.json", "test_spec", "train"])

        call_args = mock_run.call_args
        env = call_args[1]["env"]
        assert env["SPEC_VARS"] == ""

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_dstack_apply_command_structure(self, mock_run):
        """Test the dstack apply command has correct structure."""
        runner.invoke(app, ["run-remote", "spec.json", "test_spec", "train"])

        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "uv"
        assert cmd[1] == "run"
        assert cmd[2] == "--no-default-groups"  # minimal install mode
        assert cmd[3] == "dstack"
        assert cmd[4] == "apply"
        assert "-y" in cmd
        assert "--detach" in cmd


class TestRunRemoteCli:
    """Tests for run-remote CLI command."""

    def test_run_remote_help(self):
        """Test run-remote help shows all options."""
        result = runner.invoke(app, ["run-remote", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "Launch experiment on dstack" in output
        assert "SPEC_FILE" in output
        assert "SPEC_NAME" in output
        assert "--name" in output
        assert "--gpu" in output
        assert "--set" in output

    def test_run_remote_missing_args(self):
        """Test run-remote fails without required args."""
        result = runner.invoke(app, ["run-remote"])
        assert result.exit_code != 0

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_run_remote_via_cli(self, mock_run):
        """Test run-remote command invokes correctly via CLI."""
        runner.invoke(
            app,
            ["run-remote", "spec.json", "test_spec", "train"],
        )
        mock_run.assert_called_once()

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_run_remote_with_gpu_via_cli(self, mock_run):
        """Test run-remote with --gpu flag via CLI."""
        runner.invoke(
            app,
            ["run-remote", "--gpu", "spec.json", "test_spec", "train"],
        )
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert ".dstack/run-gpu-train.yml" in cmd

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_run_remote_with_sets_via_cli(self, mock_run):
        """Test run-remote with -s options via CLI."""
        runner.invoke(
            app,
            [
                "run-remote",
                "spec.json",
                "test_spec",
                "train",
                "-s",
                "env=CartPole-v1",
                "-s",
                "lr=0.001",
            ],
        )
        mock_run.assert_called_once()
        env = mock_run.call_args[1]["env"]
        assert "env=CartPole-v1" in env["SPEC_VARS"]
        assert "lr=0.001" in env["SPEC_VARS"]

    @patch("slm_lab.cli.remote.subprocess.run")
    def test_run_remote_propagates_error_code(self, mock_run):
        """Test that non-zero exit codes from dstack are propagated."""
        from unittest.mock import MagicMock
        mock_run.return_value = MagicMock(returncode=1)
        result = runner.invoke(
            app,
            ["run-remote", "spec.json", "test_spec", "train"],
        )
        assert result.exit_code == 1
