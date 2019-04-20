from slm_lab.env.vec_env import make_gym_venv
import numpy as np
import pytest


@pytest.mark.parametrize('name,state_shape', [
    ('PongNoFrameskip-v4', (84, 84)),
    # ('LunarLander-v2', (32,)),
    # ('CartPole-v0', (16,)),
])
@pytest.mark.parametrize('num_envs', (1, 4))
def test_make_gym_venv(name, state_shape, num_envs):
    seed = 0
    stack_len = 4
    venv = make_gym_venv(name, seed, stack_len, num_envs)
    venv.reset()
    for i in range(5):
        state, reward, done, _info = venv.step([venv.action_space.sample()] * num_envs)

    assert isinstance(state, np.ndarray)
    assert state.shape == (num_envs, stack_len) + state_shape
    assert isinstance(reward, np.ndarray)
    assert reward.shape == (num_envs,)
    assert isinstance(done, np.ndarray)
    assert done.shape == (num_envs,)
    assert len(_info) == num_envs
    venv.close()

# Classic dont adapt well
# test stack_len None
