from slm_lab.env.vec_env import make_gym_venv
import numpy as np
import pytest


@pytest.mark.parametrize('name,state_shape,reward_scale', [
    ('PongNoFrameskip-v4', (1, 84, 84), 'sign'),
    ('LunarLander-v2', (8,), None),
    ('CartPole-v0', (4,), None),
])
@pytest.mark.parametrize('num_envs', (1, 4))
def test_make_gym_venv_nostack(name, num_envs, state_shape, reward_scale):
    seed = 0
    frame_op = None
    frame_op_len = None
    venv = make_gym_venv(name, num_envs, seed, frame_op, frame_op_len, reward_scale)
    venv.reset()
    for i in range(5):
        state, reward, done, info = venv.step([venv.action_space.sample()] * num_envs)

    assert isinstance(state, np.ndarray)
    assert state.shape == (num_envs,) + state_shape
    assert isinstance(reward, np.ndarray)
    assert reward.shape == (num_envs,)
    assert isinstance(done, np.ndarray)
    assert done.shape == (num_envs,)
    assert len(info) == num_envs
    venv.close()


@pytest.mark.parametrize('name,state_shape, reward_scale', [
    ('PongNoFrameskip-v4', (1, 84, 84), 'sign'),
    ('LunarLander-v2', (8,), None),
    ('CartPole-v0', (4,), None),
])
@pytest.mark.parametrize('num_envs', (1, 4))
def test_make_gym_concat(name, num_envs, state_shape, reward_scale):
    seed = 0
    frame_op = 'concat'  # used for image, or for concat vector
    frame_op_len = 4
    venv = make_gym_venv(name, num_envs, seed, frame_op, frame_op_len, reward_scale)
    venv.reset()
    for i in range(5):
        state, reward, done, info = venv.step([venv.action_space.sample()] * num_envs)

    assert isinstance(state, np.ndarray)
    stack_shape = (num_envs, frame_op_len * state_shape[0],) + state_shape[1:]
    assert state.shape == stack_shape
    assert isinstance(reward, np.ndarray)
    assert reward.shape == (num_envs,)
    assert isinstance(done, np.ndarray)
    assert done.shape == (num_envs,)
    assert len(info) == num_envs
    venv.close()


@pytest.mark.skip(reason='Not implemented yet')
@pytest.mark.parametrize('name,state_shape,reward_scale', [
    ('LunarLander-v2', (8,), None),
    ('CartPole-v0', (4,), None),
])
@pytest.mark.parametrize('num_envs', (1, 4))
def test_make_gym_stack(name, num_envs, state_shape, reward_scale):
    seed = 0
    frame_op = 'stack'  # used for rnn
    frame_op_len = 4
    venv = make_gym_venv(name, num_envs, seed, frame_op, frame_op_len, reward_scale)
    venv.reset()
    for i in range(5):
        state, reward, done, info = venv.step([venv.action_space.sample()] * num_envs)

    assert isinstance(state, np.ndarray)
    stack_shape = (num_envs, frame_op_len,) + state_shape
    assert state.shape == stack_shape
    assert isinstance(reward, np.ndarray)
    assert reward.shape == (num_envs,)
    assert isinstance(done, np.ndarray)
    assert done.shape == (num_envs,)
    assert len(info) == num_envs
    venv.close()
