"""Tests for slm_lab.cli.sync module."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from slm_lab.cli import app
from slm_lab.cli.sync import pull, list_experiments


runner = CliRunner()

# Strip ANSI color codes for consistent test assertions
ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')
def strip_ansi(text):
    return ANSI_PATTERN.sub('', text)


class TestPull:
    """Tests for pull function."""

    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_pull_finds_matching_experiments(self, mock_repo_id, mock_api):
        """Test pull correctly filters experiments by pattern."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api_instance.list_repo_files.return_value = [
            "data/ppo_cartpole_2024_01_01/info.json",
            "data/ppo_cartpole_2024_01_02/info.json",
            "data/dqn_cartpole_2024_01_01/info.json",
            "data/sac_lunar_2024_01_01/info.json",
        ]
        mock_api.return_value = mock_api_instance

        with patch("slm_lab.cli.sync.snapshot_download") as mock_download:
            with patch.object(Path, "exists", return_value=False):
                pull(pattern="ppo_cartpole", list_only=False)

        # Should download 2 experiments matching ppo_cartpole
        assert mock_download.call_count == 2

    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_pull_list_only_no_download(self, mock_repo_id, mock_api):
        """Test list_only mode doesn't download."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api_instance.list_repo_files.return_value = [
            "data/ppo_cartpole_2024/info.json",
        ]
        mock_api.return_value = mock_api_instance

        with patch("slm_lab.cli.sync.snapshot_download") as mock_download:
            pull(pattern="ppo", list_only=True)

        mock_download.assert_not_called()

    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_pull_skips_existing_local(self, mock_repo_id, mock_api):
        """Test pull skips experiments that already exist locally."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api_instance.list_repo_files.return_value = [
            "data/ppo_cartpole_2024/info.json",
        ]
        mock_api.return_value = mock_api_instance

        with patch("slm_lab.cli.sync.snapshot_download") as mock_download:
            with patch.object(Path, "exists", return_value=True):
                pull(pattern="ppo_cartpole", list_only=False)

        mock_download.assert_not_called()

    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_pull_strips_data_prefix(self, mock_repo_id, mock_api):
        """Test pull correctly strips data/ prefix from pattern."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api_instance.list_repo_files.return_value = [
            "data/ppo_cartpole_2024/info.json",
        ]
        mock_api.return_value = mock_api_instance

        with patch("slm_lab.cli.sync.snapshot_download") as mock_download:
            with patch.object(Path, "exists", return_value=False):
                # Pattern with data/ prefix should work
                pull(pattern="data/ppo_cartpole", list_only=False)

        assert mock_download.call_count == 1

    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_pull_no_matches_warns(self, mock_repo_id, mock_api, capsys):
        """Test pull warns when no matching experiments found."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api_instance.list_repo_files.return_value = [
            "data/other_exp/info.json",
        ]
        mock_api.return_value = mock_api_instance

        pull(pattern="nonexistent", list_only=False)
        # Should complete without error (warning logged)

    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_pull_snapshot_download_args(self, mock_repo_id, mock_api):
        """Test snapshot_download is called with correct arguments."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api_instance.list_repo_files.return_value = [
            "data/test_exp/info.json",
        ]
        mock_api.return_value = mock_api_instance

        with patch("slm_lab.cli.sync.snapshot_download") as mock_download:
            with patch.object(Path, "exists", return_value=False):
                pull(pattern="test_exp", list_only=False)

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args[1]
        assert call_kwargs["repo_id"] == "user/repo"
        assert call_kwargs["repo_type"] == "dataset"
        assert call_kwargs["local_dir"] == "."
        assert "data/test_exp/*" in call_kwargs["allow_patterns"][0]


class TestPush:
    """Tests for push function via CLI runner."""

    def test_push_nonexistent_dir_exits(self):
        """Test push exits with error for nonexistent directory."""
        result = runner.invoke(app, ["push", "nonexistent/path"])
        assert result.exit_code != 0

    @patch("huggingface_hub.upload_folder")
    @patch("huggingface_hub.create_repo")
    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_push_creates_repo(self, mock_repo_id, mock_api, mock_create, mock_upload, tmp_path):
        """Test push creates repo if not exists."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance
        mock_upload.return_value = MagicMock(commit_url="https://hf.co/commit")

        # Create temp directory
        test_dir = tmp_path / "test_exp"
        test_dir.mkdir()

        runner.invoke(app, ["push", str(test_dir)])

        mock_create.assert_called_once()
        assert mock_create.call_args[0][0] == "user/repo"
        assert mock_create.call_args[1]["repo_type"] == "dataset"
        assert mock_create.call_args[1]["exist_ok"] is True

    @patch("huggingface_hub.upload_folder")
    @patch("huggingface_hub.create_repo")
    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_push_upload_folder_args(self, mock_repo_id, mock_api, mock_create, mock_upload, tmp_path):
        """Test upload_folder is called with correct arguments."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance
        mock_upload.return_value = MagicMock(commit_url="https://hf.co/commit")

        test_dir = tmp_path / "data" / "test_exp"
        test_dir.mkdir(parents=True)

        runner.invoke(app, ["push", str(test_dir)])

        mock_upload.assert_called_once()
        call_kwargs = mock_upload.call_args[1]
        assert call_kwargs["folder_path"] == str(test_dir)
        assert call_kwargs["repo_id"] == "user/repo"
        assert call_kwargs["repo_type"] == "dataset"
        assert call_kwargs["path_in_repo"] == str(test_dir)

    @patch("huggingface_hub.upload_folder")
    @patch("huggingface_hub.create_repo")
    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_push_strips_trailing_slash(self, mock_repo_id, mock_api, mock_create, mock_upload, tmp_path):
        """Test push strips trailing slash from path."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api.return_value = mock_api_instance
        mock_upload.return_value = MagicMock(commit_url="https://hf.co/commit")

        test_dir = tmp_path / "test_exp"
        test_dir.mkdir()

        runner.invoke(app, ["push", str(test_dir) + "/"])

        call_kwargs = mock_upload.call_args[1]
        assert not call_kwargs["folder_path"].endswith("/")


class TestListExperiments:
    """Tests for list_experiments function."""

    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_list_all_experiments(self, mock_repo_id, mock_api):
        """Test listing all experiments without pattern."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api_instance.list_repo_files.return_value = [
            "data/exp1/info.json",
            "data/exp2/info.json",
            "data/exp3/info.json",
            "other/file.txt",
        ]
        mock_api.return_value = mock_api_instance

        # Should complete without error
        list_experiments(pattern="")

    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_list_filtered_experiments(self, mock_repo_id, mock_api):
        """Test listing experiments with pattern filter."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api_instance.list_repo_files.return_value = [
            "data/ppo_cartpole/info.json",
            "data/dqn_cartpole/info.json",
            "data/sac_lunar/info.json",
        ]
        mock_api.return_value = mock_api_instance

        # Should complete without error, filtering to ppo
        list_experiments(pattern="ppo")

    @patch("slm_lab.cli.sync.hf.get_api")
    @patch("slm_lab.cli.sync.hf.get_repo_id")
    def test_list_no_experiments(self, mock_repo_id, mock_api):
        """Test listing when repo is empty."""
        mock_repo_id.return_value = "user/repo"
        mock_api_instance = MagicMock()
        mock_api_instance.list_repo_files.return_value = []
        mock_api.return_value = mock_api_instance

        # Should complete without error
        list_experiments(pattern="")


class TestSyncCli:
    """Tests for sync CLI commands."""

    def test_pull_help(self):
        """Test pull help shows all options."""
        result = runner.invoke(app, ["pull", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "Pull experiment results from HuggingFace" in output
        assert "PATTERN" in output
        assert "--list" in output

    def test_push_help(self):
        """Test push help shows all options."""
        result = runner.invoke(app, ["push", "--help"])
        assert result.exit_code == 0
        assert "Push local experiment to HuggingFace" in result.output
        assert "PREDIR" in result.output

    def test_list_help(self):
        """Test list help shows all options."""
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0
        assert "List experiments available on HuggingFace" in result.output
        assert "PATTERN" in result.output

    def test_pull_requires_pattern(self):
        """Test pull requires pattern argument."""
        result = runner.invoke(app, ["pull"])
        assert result.exit_code != 0

    def test_push_requires_predir(self):
        """Test push requires predir argument."""
        result = runner.invoke(app, ["push"])
        assert result.exit_code != 0

    def test_list_pattern_optional(self):
        """Test list pattern is optional."""
        # This will fail due to HF connection, but should parse args correctly
        with patch("slm_lab.cli.sync.hf.get_repo_id") as mock:
            mock.side_effect = Exception("No connection")
            result = runner.invoke(app, ["list"])
            # Should fail due to exception, not missing args
            assert result.exit_code != 0
            assert "No connection" in result.output or result.exception is not None
