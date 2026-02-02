import torch
import numpy as np
from slm_lab.agent.algorithm.base import Algorithm


class TestAlgorithm(Algorithm):
    """Test implementation of Algorithm for action conversion testing"""
    
    def __init__(self, is_discrete, is_venv, action_dim, num_envs=1):
        # Direct setup without using mock
        self.agent = type('Agent', (), {})()
        self.agent.env = type('Env', (), {})()
        self.agent.env.is_discrete = is_discrete
        self.agent.env.is_venv = is_venv
        self.agent.env.action_dim = action_dim
        if is_venv:
            self.agent.env.num_envs = num_envs
    
    def init_algorithm_params(self):
        pass
    
    def init_nets(self, global_nets=None):
        pass
    
    def sample(self):
        pass
    
    def train(self):
        pass
    
    def update(self):
        pass


def test_discrete_single():
    """Single discrete: (1,) → scalar int"""
    algo = TestAlgorithm(is_discrete=True, is_venv=False, action_dim=2)
    action = torch.tensor([1])
    result = algo.to_action(action)
    assert isinstance(result, (int, np.integer))
    assert result == 1


def test_discrete_vector():
    """Vector discrete: (2,) → (2,)"""
    algo = TestAlgorithm(is_discrete=True, is_venv=True, action_dim=2, num_envs=2)
    action = torch.tensor([1, 0])
    result = algo.to_action(action)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)
    np.testing.assert_array_equal(result, [1, 0])


def test_continuous_single():
    """Single continuous: (1, action_dim) → (action_dim,)"""
    algo = TestAlgorithm(is_discrete=False, is_venv=False, action_dim=4)
    action = torch.tensor([[0.5, 0.3, -0.2, 0.8]])
    result = algo.to_action(action)
    assert isinstance(result, np.ndarray)
    assert result.shape == (4,)
    np.testing.assert_array_almost_equal(result, [0.5, 0.3, -0.2, 0.8])


def test_continuous_vector():
    """Vector continuous: (num_envs, action_dim) → (num_envs, action_dim)"""
    algo = TestAlgorithm(is_discrete=False, is_venv=True, action_dim=1, num_envs=2)
    action = torch.tensor([[0.5], [0.7]])
    result = algo.to_action(action)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1)
    np.testing.assert_array_almost_equal(result, [[0.5], [0.7]])


def test_continuous_vector_reshape():
    """Vector continuous reshape: (num_envs*action_dim,) → (num_envs, action_dim)"""
    algo = TestAlgorithm(is_discrete=False, is_venv=True, action_dim=1, num_envs=2)
    action = torch.tensor([0.5, 0.7])  # Flattened
    result = algo.to_action(action)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1)
    np.testing.assert_array_almost_equal(result, [[0.5], [0.7]])