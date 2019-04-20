from slm_lab.env.wrapper import make_gym_env, LazyFrames
import numpy as np
import pytest


@pytest.mark.parametrize('name,state_shape', [
    ('PongNoFrameskip-v4', (1, 84, 84)),
    ('LunarLander-v2', (8,)),
    ('CartPole-v0', (4,)),
])
def test_make_gym_env(name, state_shape):
    seed = 0
    stack_len = 4
    env = make_gym_env(name, seed, stack_len)
    env.reset()
    for i in range(5):
        state, reward, done, info = env.step(env.action_space.sample())

    assert isinstance(state, LazyFrames)
    state = state.__array__()  # realize data
    assert isinstance(state, np.ndarray)
    if len(state_shape) == 1:
        stack_shape = (stack_len * state_shape[0],)
    else:
        stack_shape = (stack_len,) + state_shape[1:]
    assert state.shape == stack_shape
    assert state.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    env.close()


@pytest.mark.parametrize('name,state_shape', [
    ('PongNoFrameskip-v4', (1, 84, 84)),
    ('LunarLander-v2', (8,)),
    ('CartPole-v0', (4,)),
])
def test_make_gym_env_nostack(name, state_shape):
    seed = 0
    stack_len = None
    env = make_gym_env(name, seed, stack_len)
    env.reset()
    for i in range(5):
        state, reward, done, info = env.step(env.action_space.sample())

    assert isinstance(state, np.ndarray)
    assert state.shape == state_shape
    assert state.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    env.close()
