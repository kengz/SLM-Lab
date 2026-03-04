from slm_lab.experiment.control import make_agent_env
from slm_lab.spec import spec_util


def _make_sac_agent(algorithm_overrides=None):
    """Create a SAC agent with optional algorithm spec overrides."""
    spec = spec_util.get("benchmark/sac/sac_cartpole.json", "sac_cartpole")
    spec_util.tick(spec, "trial")
    spec = spec_util.override_spec(spec, "test")
    if algorithm_overrides:
        spec["agent"]["algorithm"].update(algorithm_overrides)
    agent, env = make_agent_env(spec)
    return agent


def test_sac_default_symlog_false():
    agent = _make_sac_agent()
    assert agent.algorithm.symlog is False


def test_sac_symlog_true_sets_attribute():
    agent = _make_sac_agent({"symlog": True})
    assert agent.algorithm.symlog is True
