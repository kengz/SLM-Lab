# Generic env wrappers, including for Atari/images
# They don't come with Gym but are crucial for Atari to work
# Many were adapted from OpenAI Baselines (MIT) https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
from collections import deque
from gym import spaces
from slm_lab.lib import util
import gym
import numpy as np


def try_scale_reward(cls, reward):
    '''Env class to scale reward'''
    if util.in_eval_lab_modes():  # only trigger on training
        return reward
    if cls.reward_scale is not None:
        if cls.sign_reward:
            reward = np.sign(reward)
        else:
            reward *= cls.reward_scale
    return reward


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        '''
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        '''
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        '''Do no-op action for a number of steps in [1, noop_max].'''
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class MaxAndSkipEnv(gym.Wrapper):
    '''OpenAI max-skipframe wrapper used for a NoFrameskip env'''

    def __init__(self, env, skip=4):
        '''Return only every `skip`-th frame'''
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        '''Repeat action, sum reward, and max over last observations.'''
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        '''
        Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        '''
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = info['was_real_done'] = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        '''
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        '''
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        '''Take action on reset for environments that are fixed until firing.'''
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class PreprocessImage(gym.ObservationWrapper):
    def __init__(self, env):
        '''
        Apply image preprocessing:
        - grayscale
        - downsize to 84x84
        - transpose shape from h,w,c to PyTorch format c,h,w
        '''
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, self.width, self.height), dtype=np.uint8)

    def observation(self, frame):
        return util.preprocess_image(frame)


class LazyFrames(object):
    def __init__(self, frames, frame_op='stack'):
        '''
        Wrapper to stack or concat frames by keeping unique soft reference insted of copies of data.
        So this should only be converted to numpy array before being passed to the model.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay buffers.
        @param str:frame_op 'stack' or 'concat'
        '''
        self._frames = frames
        self._out = None
        if frame_op == 'stack':
            self._frame_op = np.stack
        elif frame_op == 'concat':
            self._frame_op = np.concatenate
        else:
            raise ValueError('frame_op not recognized for LazyFrames. Choose from "stack", "concat"')

    def _force(self):
        if self._out is None:
            self._out = self._frame_op(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def astype(self, dtype):
        '''To prevent state.astype(np.float16) breaking on LazyFrames'''
        return self


class FrameStack(gym.Wrapper):
    def __init__(self, env, frame_op, frame_op_len):
        '''
        Stack/concat last k frames. Returns lazy array, which is much more memory efficient.
        @param str:frame_op 'concat' or 'stack'. Note: use concat for image since the shape is (1, 84, 84) concat-able.
        @param int:frame_op_len The number of frames to keep for frame_op
        '''
        gym.Wrapper.__init__(self, env)
        self.frame_op = frame_op
        self.frame_op_len = frame_op_len
        self.frames = deque([], maxlen=self.frame_op_len)
        old_shape = env.observation_space.shape
        if self.frame_op == 'concat':  # concat multiplies first dim
            shape = (self.frame_op_len * old_shape[0],) + old_shape[1:]
        elif self.frame_op == 'stack':  # stack creates new dim
            shape = (self.frame_op_len,) + old_shape
        else:
            raise ValueError('frame_op not recognized for FrameStack. Choose from "stack", "concat".')
        self.observation_space = spaces.Box(
            low=np.min(env.observation_space.low),
            high=np.max(env.observation_space.high),
            shape=shape, dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.frame_op_len):
            self.frames.append(ob.astype(np.float16))
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob.astype(np.float16))
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.frame_op_len
        return LazyFrames(list(self.frames), self.frame_op)


class UnityVecFrameStack(gym.Wrapper):
    '''Frame stack wrapper for Unity vector environment'''

    def __init__(self, env, frame_op, frame_op_len):
        self.env = env
        assert frame_op in ('concat', 'stack'), 'Invalid frame_op mode'
        self.is_stack = frame_op == 'stack'
        self.frame_op_len = frame_op_len
        self.spec = env.spec
        wos = env.observation_space  # wrapped ob space
        if self.is_stack:
            self.shape_dim0 = 1
            low = np.repeat(np.expand_dims(wos.low, axis=0), self.frame_op_len, axis=0)
            high = np.repeat(np.expand_dims(wos.high, axis=0), self.frame_op_len, axis=0)
        else:  # concat
            self.shape_dim0 = wos.shape[0]
            low = np.repeat(wos.low, self.frame_op_len, axis=0)
            high = np.repeat(wos.high, self.frame_op_len, axis=0)
        self.stackedobs = np.zeros((env.num_envs,) + low.shape, low.dtype)
        self.observation_space = spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)
        self.action_space = env.action_space

    def step(self, action):
        obs, rews, news, infos = self.env.step(action)
        self.stackedobs[:, :-self.shape_dim0] = self.stackedobs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        if self.is_stack:
            obs = np.expand_dims(obs, axis=1)
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs.copy(), rews, news, infos

    def reset(self):
        obs = self.env.reset()
        self.stackedobs[...] = 0
        if self.is_stack:
            obs = np.expand_dims(obs, axis=1)
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs.copy()


class NormalizeStateEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        '''
        Normalize observations on-line
        Adapted from https://github.com/ikostrikov/pytorch-a3c/blob/e898f7514a03de73a2bf01e7b0f17a6f93963389/envs.py (MIT)
        '''
        super().__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class ScaleRewardEnv(gym.RewardWrapper):
    def __init__(self, env, reward_scale):
        '''
        Rescale reward
        @param (str,float):reward_scale If 'sign', use np.sign, else multiply with the specified float scale
        '''
        gym.Wrapper.__init__(self, env)
        self.reward_scale = reward_scale
        self.sign_reward = self.reward_scale == 'sign'

    def reward(self, reward):
        return try_scale_reward(self, reward)


class TrackReward(gym.Wrapper):
    def __init__(self, env):
        '''
        Self-tracking as a simple solution to total reward tracking
        Tracks the latest episodic rewards
        '''
        gym.Wrapper.__init__(self, env)
        self.tracked_reward = 0
        self.total_reward = np.nan

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.tracked_reward += reward
        # use self.was_real_done from EpisodicLifeEnv, or plain done
        real_done = info.get('was_real_done', False) or done
        not_real_done = (1 - real_done)
        # update total_reward only when real_done, else use old value
        self.total_reward = np.nan_to_num(self.total_reward) * not_real_done + self.tracked_reward * real_done
        # reset to 0 on real_done, i.e. multiply with not_real_done
        self.tracked_reward = self.tracked_reward * not_real_done
        info.update({'total_reward': self.total_reward})
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def wrap_atari(env):
    '''Apply a common set of wrappers for Atari games'''
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(env, episode_life=True, stack_len=None):
    '''Wrap Atari environment DeepMind-style'''
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = PreprocessImage(env)
    if stack_len is not None:  # use concat for image (1, 84, 84)
        env = FrameStack(env, 'concat', stack_len)
    return env


def make_gym_env(name, seed=None, frame_op=None, frame_op_len=None, reward_scale=None, normalize_state=False, episode_life=True):
    '''General method to create any Gym env; auto wraps Atari'''
    env = gym.make(name)
    if seed is not None:
        env.seed(seed)
    if 'NoFrameskip' in env.spec.id:  # Atari
        env = wrap_atari(env)
        # no reward clipping to allow monitoring; Atari memory clips it
        env = wrap_deepmind(env, episode_life, frame_op_len)
    elif len(env.observation_space.shape) == 3:  # image-state env
        env = PreprocessImage(env)
        if normalize_state:
            env = NormalizeStateEnv(env)
        if frame_op_len is not None:  # use concat for image (1, 84, 84)
            env = FrameStack(env, 'concat', frame_op_len)
    else:  # vector-state env
        if normalize_state:
            env = NormalizeStateEnv(env)
        if frame_op is not None:
            Stacker = UnityVecFrameStack if name.startswith('Unity') else FrameStack
            env = Stacker(env, frame_op, frame_op_len)
    env = TrackReward(env)  # auto-track total reward
    if reward_scale is not None:
        env = ScaleRewardEnv(env, reward_scale)
    return env
