"""Tests for search module."""
import glob
import os
import pandas as pd
import pytest
import tempfile
import shutil

from slm_lab.experiment.search import cleanup_trial_models
from slm_lab.spec import spec_util


@pytest.fixture
def search_spec():
    """Create a minimal spec for search testing."""
    spec = spec_util.get('demo.json', 'ppo_cartpole')
    spec_util.tick(spec, 'experiment')
    return spec


@pytest.fixture
def mock_experiment_df():
    """Create a mock experiment_df with trial results."""
    return pd.DataFrame({
        'trial': [0, 1, 2, 3, 4],
        'total_reward_ma': [100.0, 50.0, 150.0, 75.0, 25.0],
    })


@pytest.fixture
def model_dir_with_files(search_spec):
    """Create a temporary model directory with mock model files."""
    from slm_lab.lib import util

    predir = util.get_predir(search_spec)
    model_dir = f'{predir}/model'
    os.makedirs(model_dir, exist_ok=True)

    spec_name = search_spec['name']

    # Create mock model files for 5 trials (t0-t4), each with 2 sessions (s0, s1)
    for trial in range(5):
        for session in range(2):
            for net in ['actor', 'critic']:
                # Create files matching the actual naming pattern
                model_file = f'{model_dir}/{spec_name}_t{trial}_s{session}_{net}_model.pt'
                optim_file = f'{model_dir}/{spec_name}_t{trial}_s{session}_{net}_optim.pt'
                # Create empty files
                open(model_file, 'w').close()
                open(optim_file, 'w').close()

    yield model_dir

    # Cleanup after test
    if os.path.exists(predir):
        shutil.rmtree(predir)


def test_cleanup_trial_models_removes_bottom_trials(search_spec, mock_experiment_df, model_dir_with_files):
    """Test that cleanup removes model files for trials not in top N."""
    # Before cleanup: should have 5 trials * 2 sessions * 2 nets * 2 files (model+optim) = 40 files
    files_before = glob.glob(f'{model_dir_with_files}/*.pt')
    assert len(files_before) == 40, f"Expected 40 files before cleanup, got {len(files_before)}"

    # Run cleanup with keep_top_n=3
    # Top 3 by total_reward_ma: trial 2 (150), trial 0 (100), trial 3 (75)
    # Removed: trial 1 (50), trial 4 (25)
    cleanup_trial_models(search_spec, mock_experiment_df, keep_top_n=3)

    # After cleanup: should have 3 trials * 2 sessions * 2 nets * 2 files = 24 files
    files_after = glob.glob(f'{model_dir_with_files}/*.pt')
    assert len(files_after) == 24, f"Expected 24 files after cleanup, got {len(files_after)}"

    # Verify correct trials were kept (0, 2, 3)
    spec_name = search_spec['name']
    for trial in [0, 2, 3]:
        trial_files = glob.glob(f'{model_dir_with_files}/{spec_name}_t{trial}_*.pt')
        assert len(trial_files) == 8, f"Trial {trial} should have 8 files, got {len(trial_files)}"

    # Verify correct trials were removed (1, 4)
    for trial in [1, 4]:
        trial_files = glob.glob(f'{model_dir_with_files}/{spec_name}_t{trial}_*.pt')
        assert len(trial_files) == 0, f"Trial {trial} should have 0 files, got {len(trial_files)}"


def test_cleanup_trial_models_keep_all_when_few_trials(search_spec, model_dir_with_files):
    """Test that cleanup keeps all trials when total <= keep_top_n."""
    # Only 2 trials in experiment_df
    df = pd.DataFrame({
        'trial': [0, 1],
        'total_reward_ma': [100.0, 50.0],
    })

    files_before = glob.glob(f'{model_dir_with_files}/*.pt')

    # With keep_top_n=3 and only 2 trials, nothing should be removed
    cleanup_trial_models(search_spec, df, keep_top_n=3)

    files_after = glob.glob(f'{model_dir_with_files}/*.pt')
    # Files for trials not in df (2, 3, 4) should still exist
    assert len(files_after) == len(files_before)


def test_cleanup_trial_models_no_model_dir(search_spec, mock_experiment_df):
    """Test that cleanup handles missing model directory gracefully."""
    from slm_lab.lib import util

    predir = util.get_predir(search_spec)
    model_dir = f'{predir}/model'

    # Ensure model directory doesn't exist
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # Should not raise, just log info
    cleanup_trial_models(search_spec, mock_experiment_df, keep_top_n=3)

    # Cleanup
    if os.path.exists(predir):
        shutil.rmtree(predir)
