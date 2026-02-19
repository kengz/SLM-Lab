import pytest
from slm_lab.spec import spec_util


def test_yaml_spec_loading():
    """Test that a YAML spec file loads correctly via spec_util.get()"""
    spec = spec_util.get("benchmark/ppo/ppo_cartpole.yaml", "ppo_cartpole")
    assert spec is not None
    assert spec["name"] == "ppo_cartpole"
    assert spec["agent"]["name"] == "PPO"


def test_yaml_spec_check():
    """Test that spec check works on YAML-loaded specs"""
    spec = spec_util.get("benchmark/ppo/ppo_cartpole.yaml", "ppo_cartpole")
    assert spec_util.check(spec)


def test_yaml_matches_json():
    """Test that YAML and JSON specs produce equivalent configurations"""
    json_spec = spec_util.get("benchmark/ppo/ppo_cartpole.json", "ppo_cartpole")
    yaml_spec = spec_util.get("benchmark/ppo/ppo_cartpole.yaml", "ppo_cartpole")
    # Compare agent, env sections (meta has runtime-generated fields)
    assert json_spec["agent"] == yaml_spec["agent"]
    assert json_spec["env"] == yaml_spec["env"]
    for key in (
        "distributed",
        "log_frequency",
        "eval_frequency",
        "max_session",
        "max_trial",
    ):
        assert json_spec["meta"][key] == yaml_spec["meta"][key]


def test_json_backward_compat():
    """Ensure JSON specs still work after YAML support added"""
    spec = spec_util.get("benchmark/ppo/ppo_cartpole.json", "ppo_cartpole")
    assert spec is not None
    assert spec["name"] == "ppo_cartpole"
    assert spec["agent"]["name"] == "PPO"
