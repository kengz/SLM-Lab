from slm_lab.experiment.control import make_agent_env
from slm_lab.spec import spec_util


def _make_ppo_agent(algorithm_overrides=None):
    """Create a PPO agent with optional algorithm spec overrides."""
    spec = spec_util.get("benchmark/ppo/ppo_cartpole.json", "ppo_cartpole")
    spec_util.tick(spec, "trial")
    spec = spec_util.override_spec(spec, "test")
    if algorithm_overrides:
        spec["agent"]["algorithm"].update(algorithm_overrides)
    agent, env = make_agent_env(spec)
    return agent


def test_ppo_default_symlog_false():
    agent = _make_ppo_agent()
    assert agent.algorithm.symlog is False


def test_ppo_default_normalize_advantages_standardize():
    agent = _make_ppo_agent()
    assert agent.algorithm.normalize_advantages == "standardize"


def test_ppo_symlog_true_sets_attribute():
    agent = _make_ppo_agent({"symlog": True})
    assert agent.algorithm.symlog is True


def test_ppo_percentile_normalizer_created():
    agent = _make_ppo_agent({"normalize_advantages": "percentile"})
    assert agent.algorithm.normalize_advantages == "percentile"
    assert hasattr(agent.algorithm, "percentile_normalizer")
