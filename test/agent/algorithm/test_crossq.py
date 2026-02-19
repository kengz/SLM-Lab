"""Tests for CrossQ algorithm (Tier 2)."""

import pytest
import torch
import torch.nn as nn

from slm_lab.agent.algorithm.crossq import CrossQ
from slm_lab.agent.algorithm.sac import SoftActorCritic

try:
    import torcharc

    HAS_TORCHARC = True
except ImportError:
    HAS_TORCHARC = False


# ---------------------------------------------------------------------------
# Unit tests — no env / agent needed
# ---------------------------------------------------------------------------


class TestCrossQClass:
    def test_inherits_from_sac(self):
        assert issubclass(CrossQ, SoftActorCritic)

    def test_class_exists_in_algorithm_module(self):
        from slm_lab.agent import algorithm

        assert hasattr(algorithm, "CrossQ")


class TestCalcQCross:
    """Test calc_q_cross with a simple Linear Q-net (no torcharc needed)."""

    @pytest.fixture
    def q_net(self):
        """Simple Q-net: input_dim=6 (state=4 + action=2), output=1."""
        net = nn.Linear(6, 1)
        return net

    def test_output_shapes(self, q_net):
        batch = 8
        states = torch.randn(batch, 4)
        actions = torch.randn(batch, 2)
        next_states = torch.randn(batch, 4)
        next_actions = torch.randn(batch, 2)

        # Call unbound — CrossQ.calc_q_cross is a regular method, invoke via class
        q_current, q_next = CrossQ.calc_q_cross(
            None, states, actions, next_states, next_actions, q_net
        )
        assert q_current.shape == (batch,)
        assert q_next.shape == (batch,)

    def test_batch_split_correctness(self, q_net):
        """Verify the first half of the concatenated batch corresponds to current."""
        batch = 4
        states = torch.ones(batch, 4)
        actions = torch.ones(batch, 2)
        next_states = torch.zeros(batch, 4)
        next_actions = torch.zeros(batch, 2)

        q_current, q_next = CrossQ.calc_q_cross(
            None, states, actions, next_states, next_actions, q_net
        )
        # current and next should differ because inputs differ
        assert not torch.allclose(q_current, q_next)

    def test_same_input_gives_same_output(self, q_net):
        """When current == next, both halves should be identical."""
        batch = 4
        states = torch.randn(batch, 4)
        actions = torch.randn(batch, 2)

        q_current, q_next = CrossQ.calc_q_cross(
            None, states, actions, states, actions, q_net
        )
        assert torch.allclose(q_current, q_next)


class TestCalcQCrossDiscrete:
    """Test calc_q_cross_discrete with a simple Linear Q-net."""

    @pytest.fixture
    def q_net(self):
        """Simple Q-net: input=state_dim=4, output=action_dim=2."""
        return nn.Linear(4, 2)

    def test_output_shapes(self, q_net):
        batch = 8
        states = torch.randn(batch, 4)
        next_states = torch.randn(batch, 4)

        q_current, q_next = CrossQ.calc_q_cross_discrete(
            None, states, next_states, q_net
        )
        assert q_current.shape == (batch, 2)
        assert q_next.shape == (batch, 2)

    def test_same_input_gives_same_output(self, q_net):
        batch = 4
        states = torch.randn(batch, 4)

        q_current, q_next = CrossQ.calc_q_cross_discrete(None, states, states, q_net)
        assert torch.allclose(q_current, q_next)


class TestUpdateNetsNoop:
    def test_update_nets_is_noop(self):
        """update_nets should do nothing (no target networks)."""
        crossq = CrossQ.__new__(CrossQ)
        # Should not raise
        crossq.update_nets()


# ---------------------------------------------------------------------------
# Integration tests — require agent + env via spec
# ---------------------------------------------------------------------------


def _get_crossq_cartpole_spec():
    """Build a minimal CrossQ spec for CartPole (discrete) using MLPNet."""
    from slm_lab.spec import spec_util

    spec = spec_util.get("benchmark/sac/sac_cartpole.json", "sac_cartpole")
    # Override to CrossQ
    spec["agent"]["name"] = "CrossQ"
    spec["agent"]["algorithm"]["name"] = "CrossQ"
    spec["agent"]["algorithm"]["training_iter"] = 1  # UTD=1
    spec = spec_util.override_spec(spec, "test")
    return spec


def _get_crossq_pendulum_spec():
    """Build a minimal CrossQ spec for Pendulum (continuous) using MLPNet."""
    from slm_lab.spec import spec_util

    spec = spec_util.get("benchmark/sac/sac_cartpole.json", "sac_cartpole")
    # Override to CrossQ + continuous env
    spec["agent"]["name"] = "CrossQ"
    spec["agent"]["algorithm"]["name"] = "CrossQ"
    spec["agent"]["algorithm"]["action_pdtype"] = "default"
    spec["agent"]["algorithm"]["training_iter"] = 1
    spec["env"]["name"] = "Pendulum-v1"
    spec = spec_util.override_spec(spec, "test")
    return spec


class TestCrossQIntegration:
    def test_no_target_networks(self):
        from slm_lab.experiment.control import make_agent_env
        from slm_lab.spec import spec_util

        spec = _get_crossq_cartpole_spec()
        spec_util.tick(spec, "trial")
        agent, env = make_agent_env(spec)
        algo = agent.algorithm

        assert not hasattr(algo, "target_q1_net")
        assert not hasattr(algo, "target_q2_net")

    def test_net_names(self):
        from slm_lab.experiment.control import make_agent_env
        from slm_lab.spec import spec_util

        spec = _get_crossq_cartpole_spec()
        spec_util.tick(spec, "trial")
        agent, env = make_agent_env(spec)

        assert agent.algorithm.net_names == ["net", "q1_net", "q2_net"]

    def test_session_cartpole(self):
        """CrossQ completes a short training session on CartPole."""
        from slm_lab.experiment.control import Session
        from slm_lab.spec import spec_util

        spec = _get_crossq_cartpole_spec()
        spec_util.tick(spec, "trial")
        spec_util.tick(spec, "session")
        spec_util.save(spec, unit="trial")
        session = Session(spec)
        metrics = session.run()
        assert isinstance(metrics, dict)

    def test_session_pendulum(self):
        """CrossQ completes a short training session on Pendulum (continuous)."""
        from slm_lab.experiment.control import Session
        from slm_lab.spec import spec_util

        spec = _get_crossq_pendulum_spec()
        spec_util.tick(spec, "trial")
        spec_util.tick(spec, "session")
        spec_util.save(spec, unit="trial")
        session = Session(spec)
        metrics = session.run()
        assert isinstance(metrics, dict)

    def test_bn_mode_switching(self):
        """Critics switch between eval (target) and train (cross forward) modes."""
        from slm_lab.experiment.control import make_agent_env
        from slm_lab.spec import spec_util

        spec = _get_crossq_cartpole_spec()
        spec_util.tick(spec, "trial")
        agent, env = make_agent_env(spec)
        algo = agent.algorithm

        # After init, nets are in train mode
        assert algo.q1_net.training
        assert algo.q2_net.training

        # Simulate eval mode switch (as in train() target computation)
        algo.q1_net.eval()
        algo.q2_net.eval()
        assert not algo.q1_net.training
        assert not algo.q2_net.training

        # Switch back to train (as in cross batch norm forward)
        algo.q1_net.train()
        algo.q2_net.train()
        assert algo.q1_net.training
        assert algo.q2_net.training
