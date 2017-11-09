'''
The environment module
Contains graduated components from experiments for building/using environment.
Provides the rich experience for agent embodiment,
reflects the curriculum and allows teaching (possibly allows teacher to enter).
To be designed by human and evolution module,
based on the curriculum and fitness metrics.
'''
import os
from slm_lab.lib import util


def get_env_path(env_name):
    env_path = util.smart_path(
        f'node_modules/slm-env-{env_name}/build/{env_name}')
    env_dir = os.path.dirname(env_path)
    assert os.path.exists(
        env_dir), f'Missing {env_path}. See README to install from yarn.'
    return env_path
