# inspired by nsavinov/gym-vizdoom and ppaquette/gym-doom
import numpy as np
import gym.spaces as spaces
from gym import Env
from gym.envs.classic_control import rendering
from vizdoom import DoomGame


class VizDoomEnv(Env):
    """
    Wrapper for vizdoom to use as an OpenAI gym environment.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, cfg_name, repeat=1):
        super(VizDoomEnv, self).__init__()
        self.game = DoomGame()
        self.game.load_config('./slm_lab/env/vizdoom/cfgs/' + cfg_name + '.cfg')
        self._viewer = None
        self.repeat = 1
        # In future, need to update action to handle (continuous) DELTA buttons using gym's Box space
        self.action_space = spaces.MultiDiscrete([2] * self.game.get_available_buttons_size())
        self.action_space.dtype = 'uint8'
        output_shape = (self.game.get_screen_height(), self.game.get_screen_width(), self.game.get_screen_channels())
        self.observation_space = spaces.Box(low=0, high=255, shape=output_shape, dtype='uint8')
        self.game.init()

    def close(self):
        self.game.close()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def seed(self, seed=None):
        self.game.set_seed(seed)

    def step(self, action):
        reward = self.game.make_action(list(action), self.repeat)
        state = self.game.get_state()
        done = self.game.is_episode_finished()
        # info = self._get_game_variables(state.game_variables)
        info = {}
        if state is not None:
            observation = state.screen_buffer.transpose(1, 2, 0)
        else:
            observation = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
        return observation, reward, done, info

    def reset(self):
        # self.seed(seed)
        self.game.new_episode()
        return self.game.get_state().screen_buffer.transpose(1, 2, 0)

    def render(self, mode='human', close=False):
        if close:
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            return
        img = None
        state = self.game.get_state()
        if state is not None:
            img = state.screen_buffer
        if img is None:
            # at the end of the episode
            img = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)
        if mode == 'rgb_array':
            return img
        elif mode is 'human':
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(img.transpose(1, 2, 0))

    def _get_game_variables(self, state_variables):
        info = {}
        if state_variables is not None:
            info['KILLCOUNT'] = state_variables[0]
            info['ITEMCOUNT'] = state_variables[1]
            info['SECRETCOUNT'] = state_variables[2]
            info['FRAGCOUNT'] = state_variables[3]
            info['HEALTH'] = state_variables[4]
            info['ARMOR'] = state_variables[5]
            info['DEAD'] = state_variables[6]
            info['ON_GROUND'] = state_variables[7]
            info['ATTACK_READY'] = state_variables[8]
            info['ALTATTACK_READY'] = state_variables[9]
            info['SELECTED_WEAPON'] = state_variables[10]
            info['SELECTED_WEAPON_AMMO'] = state_variables[11]
            info['AMMO1'] = state_variables[12]
            info['AMMO2'] = state_variables[13]
            info['AMMO3'] = state_variables[14]
            info['AMMO4'] = state_variables[15]
            info['AMMO5'] = state_variables[16]
            info['AMMO6'] = state_variables[17]
            info['AMMO7'] = state_variables[18]
            info['AMMO8'] = state_variables[19]
            info['AMMO9'] = state_variables[20]
            info['AMMO0'] = state_variables[21]
        return info
