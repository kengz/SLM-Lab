from slm_lab.env.wrapper import make_gym_env, LazyFrames
import numpy as np
import pytest


@pytest.mark.parametrize('name,state_shape,reward_scale', [
    ('PongNoFrameskip-v4', (1, 84, 84), 'sign'),
    ('LunarLander-v2', (8,), None),
    ('CartPole-v0', (4,), None),
])
def test_make_gym_env_nostack(name, state_shape, reward_scale):
    seed = 0
    frame_op = None
    frame_op_len = None
    env = make_gym_env(name, seed, frame_op, frame_op_len, reward_scale)
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


@pytest.mark.parametrize('name,state_shape,reward_scale', [
    ('PongNoFrameskip-v4', (1, 84, 84), 'sign'),
    ('LunarLander-v2', (8,), None),
    ('CartPole-v0', (4,), None),
])
def test_make_gym_env_concat(name, state_shape, reward_scale):
    seed = 0
    frame_op = 'concat'  # used for image, or for concat vector
    frame_op_len = 4
    env = make_gym_env(name, seed, frame_op, frame_op_len, reward_scale)
    env.reset()
    for i in range(5):
        state, reward, done, info = env.step(env.action_space.sample())

    assert isinstance(state, LazyFrames)
    state = state.__array__()  # realize data
    assert isinstance(state, np.ndarray)
    # concat multiplies first dim
    stack_shape = (frame_op_len * state_shape[0],) + state_shape[1:]
    assert state.shape == stack_shape
    assert state.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    env.close()


@pytest.mark.parametrize('name,state_shape, reward_scale', [
    ('LunarLander-v2', (8,), None),
    ('CartPole-v0', (4,), None),
])
def test_make_gym_env_stack(name, state_shape, reward_scale):
    seed = 0
    frame_op = 'stack'  # used for rnn
    frame_op_len = 4
    env = make_gym_env(name, seed, frame_op, frame_op_len, reward_scale)
    env.reset()
    for i in range(5):
        state, reward, done, info = env.step(env.action_space.sample())

    assert isinstance(state, LazyFrames)
    state = state.__array__()  # realize data
    assert isinstance(state, np.ndarray)
    # stack creates new dim
    stack_shape = (frame_op_len, ) + state_shape
    assert state.shape == stack_shape
    assert state.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    env.close()
