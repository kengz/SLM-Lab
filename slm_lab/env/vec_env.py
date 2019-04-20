# Wrappers for parallel vector environments.
# Adapted from OpenAI Baselines (MIT) https://github.com/openai/baselines/tree/master/baselines/common/vec_env
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial
from gym import spaces
from slm_lab.env.wrapper import make_gym_env
from slm_lab.lib import logger
import contextlib
import ctypes
import gym
import numpy as np
import os
import torch.multiprocessing as mp


_NP_TO_CT = {
    np.float32: ctypes.c_float,
    np.int32: ctypes.c_int32,
    np.int8: ctypes.c_int8,
    np.uint8: ctypes.c_char,
    np.bool: ctypes.c_bool,
}


# helper methods


@contextlib.contextmanager
def clear_mpi_env_vars():
    '''
    from mpi4py import MPI will call MPI_Init by default.  If the child process has MPI environment variables, MPI will think that the child process is an MPI process just like the parent and do bad things such as hang.
    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting multiprocessing Processes.
    '''
    removed_environment = {}
    for k, v in list(os.environ.items()):
        for prefix in ['OMPI_', 'PMI_']:
            if k.startswith(prefix):
                removed_environment[k] = v
                del os.environ[k]
    try:
        yield
    finally:
        os.environ.update(removed_environment)


def copy_obs_dict(obs):
    '''Deep-copy an observation dict.'''
    return {k: np.copy(v) for k, v in obs.items()}


def dict_to_obs(obs_dict):
    '''Convert an observation dict into a raw array if the original observation space was not a Dict space.'''
    if set(obs_dict.keys()) == {None}:
        return obs_dict[None]
    return obs_dict


def obs_to_dict(obs):
    '''Convert an observation into a dict.'''
    if isinstance(obs, dict):
        return obs
    return {None: obs}


def obs_space_info(obs_space):
    '''
    Get dict-structured information about a gym.Space.
    @returns (keys, shapes, dtypes)
    - keys: a list of dict keys.
    - shapes: a dict mapping keys to shapes.
    - dtypes: a dict mapping keys to dtypes.
    '''
    if isinstance(obs_space, gym.spaces.Dict):
        assert isinstance(obs_space.spaces, OrderedDict)
        subspaces = obs_space.spaces
    else:
        subspaces = {None: obs_space}
    keys = []
    shapes = {}
    dtypes = {}
    for key, box in subspaces.items():
        keys.append(key)
        shapes[key] = box.shape
        dtypes[key] = box.dtype
    return keys, shapes, dtypes


def tile_images(img_nhwc):
    '''
    Tile N images into a rectangular grid for rendering

    @param img_nhwc list or array of images, with shape (batch, h, w, c)
    @returns bigim_HWc ndarray with shape (h',w',c)
    '''
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N) / H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(N, H * W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H * h, W * w, c)
    return img_Hh_Ww_c


def subproc_worker(
        pipe, parent_pipe, env_fn_wrapper,
        obs_bufs, obs_shapes, obs_dtypes, keys):
    '''
    Control a single environment instance using IPC and shared memory. Used by ShmemVecEnv.
    '''
    def _write_obs(maybe_dict_obs):
        flatdict = obs_to_dict(maybe_dict_obs)
        for k in keys:
            dst = obs_bufs[k].get_obj()
            dst_np = np.frombuffer(dst, dtype=obs_dtypes[k]).reshape(obs_shapes[k])
            np.copyto(dst_np, flatdict[k])

    env = env_fn_wrapper.x()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == 'reset':
                pipe.send(_write_obs(env.reset()))
            elif cmd == 'step':
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                pipe.send((_write_obs(obs), reward, done, info))
            elif cmd == 'render':
                pipe.send(env.render(mode='rgb_array'))
            elif cmd == 'close':
                pipe.send(None)
                break
            else:
                raise RuntimeError(f'Got unrecognized cmd {cmd}')
    except KeyboardInterrupt:
        logger.exception('ShmemVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()


# vector environment wrappers


class CloudpickleWrapper(object):
    '''
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    '''

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv(ABC):
    '''
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that each observation becomes an batch of observations, and expected action is a batch of actions to be applied per-environment.
    '''
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        '''
        Reset all the environments and return an array of observations, or a dict of observation arrays.

        If step_async is still doing work, that work will be cancelled and step_wait() should not be called until step_async() is invoked again.
        '''
        pass

    @abstractmethod
    def step_async(self, actions):
        '''
        Tell all the environments to start taking a step with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is already pending.
        '''
        pass

    @abstractmethod
    def step_wait(self):
        '''
        Wait for the step taken with step_async().

        @returns (obs, rews, dones, infos)
         - obs: an array of observations, or a dict of arrays of observations.
         - rews: an array of rewards
         - dones: an array of 'episode done' booleans
         - infos: a sequence of info objects
        '''
        pass

    def close_extras(self):
        '''
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        '''
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        '''
        Step the environments synchronously.

        This is available for backwards compatibility.
        '''
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        '''Return RGB images from each environment'''
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


class DummyVecEnv(VecEnv):
    '''
    VecEnv that does runs multiple environments sequentially, that is, the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case, avoids communication overhead)
    '''

    def __init__(self, env_fns):
        '''
        @param env_fns iterable of functions that build environments
        '''
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = {k: np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.spec = self.envs[0].spec

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.num_envs:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.num_envs == 1, f'actions {actions} is either not a list or has a wrong size - cannot match to {self.num_envs} environments'
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)
            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())

    def reset(self):
        for e in range(self.num_envs):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_obs[k][e] = obs
            else:
                self.buf_obs[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)


class VecEnvWrapper(VecEnv):
    '''
    An environment wrapper that applies to an entire batch of environments at once.
    '''

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        observation_space = observation_space or venv.observation_space
        action_space = action_space or venv.action_space
        VecEnv.__init__(self, venv.num_envs, observation_space, action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()


class ShmemVecEnv(VecEnv):
    '''
    Optimized version of SubprocVecEnv that uses shared variables to communicate observations.
    '''

    def __init__(self, env_fns, spaces=None, context='spawn'):
        '''
        If you don't specify observation_space, we'll have to create a dummy environment to get it.
        '''
        ctx = mp.get_context(context)
        if spaces:
            observation_space, action_space = spaces
        else:
            logger.info('Creating dummy env object to get spaces')
            dummy = env_fns[0]()
            observation_space, action_space = dummy.observation_space, dummy.action_space
            dummy.close()
            del dummy
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        self.obs_keys, self.obs_shapes, self.obs_dtypes = obs_space_info(observation_space)
        self.obs_bufs = [
            {k: ctx.Array(_NP_TO_CT[self.obs_dtypes[k].type], int(np.prod(self.obs_shapes[k]))) for k in self.obs_keys}
            for _ in env_fns]
        self.parent_pipes = []
        self.procs = []
        with clear_mpi_env_vars():
            for env_fn, obs_buf in zip(env_fns, self.obs_bufs):
                wrapped_fn = CloudpickleWrapper(env_fn)
                parent_pipe, child_pipe = ctx.Pipe()
                proc = ctx.Process(
                    target=subproc_worker,
                    args=(child_pipe, parent_pipe, wrapped_fn, obs_buf, self.obs_shapes, self.obs_dtypes, self.obs_keys))
                proc.daemon = True
                self.procs.append(proc)
                self.parent_pipes.append(parent_pipe)
                proc.start()
                child_pipe.close()
        self.waiting_step = False
        self.viewer = None

    def reset(self):
        if self.waiting_step:
            logger.warn('Called reset() while waiting for the step to complete')
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        return self._decode_obses([pipe.recv() for pipe in self.parent_pipes])

    def step_async(self, actions):
        assert len(actions) == len(self.parent_pipes)
        for pipe, act in zip(self.parent_pipes, actions):
            pipe.send(('step', act))

    def step_wait(self):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        obs, rews, dones, infos = zip(*outs)
        return self._decode_obses(obs), np.array(rews), np.array(dones), infos

    def close_extras(self):
        if self.waiting_step:
            self.step_wait()
        for pipe in self.parent_pipes:
            pipe.send(('close', None))
        for pipe in self.parent_pipes:
            pipe.recv()
            pipe.close()
        for proc in self.procs:
            proc.join()

    def get_images(self, mode='human'):
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        return [pipe.recv() for pipe in self.parent_pipes]

    def _decode_obses(self, obs):
        result = {}
        for k in self.obs_keys:
            bufs = [b[k] for b in self.obs_bufs]
            o = [np.frombuffer(b.get_obj(), dtype=self.obs_dtypes[k]).reshape(self.obs_shapes[k]) for b in bufs]
            result[k] = np.array(o)
        return dict_to_obs(result)


class VecFrameStack(VecEnvWrapper):
    '''Frame stack wrapper for vector environment'''

    def __init__(self, venv, k):
        self.venv = venv
        self.k = k
        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]
        low = np.repeat(wos.low, self.k, axis=0)
        high = np.repeat(wos.high, self.k, axis=0)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs[:, :-self.shape_dim0] = self.stackedobs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs


def make_gym_venv(name, seed=0, stack_len=None, num_envs=4):
    '''General method to create any parallel vectorized Gym env; auto wraps Atari'''
    venv = [
        # don't stack on individual env, but stack as vector
        partial(make_gym_env, name, seed+i, stack_len=None)
        for i in range(num_envs)
    ]
    if len(venv) > 1:
        venv = ShmemVecEnv(venv, context='fork')
    else:
        venv = DummyVecEnv(venv)
    if stack_len is not None:
        venv = VecFrameStack(venv, stack_len)
    return venv
