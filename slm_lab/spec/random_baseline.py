# module to generate random baselines
# Run as: python slm_lab/spec/random_baseline.py
from slm_lab.lib import logger, util
import gym
import numpy as np
import pydash as ps


# extra envs to include
INCLUDE_ENVS = [
    'vizdoom-v0',
]
EXCLUDE_ENVS = [
    'CarRacing-v0',  # window bug
]
NUM_EVAL = 100


def enum_envs():
    '''Enumerate all the env names of the latest version'''
    all_envs = [es.id for es in gym.envs.registration.registry.all()]
    env_dict = {}  # filter latest version: later occurence will replace
    for k in all_envs:
        name, version = k.rsplit('-', 1)
        env_dict[name] = version
    envs = [f'{k}-{v}' for k, v in env_dict.items()]
    envs += INCLUDE_ENVS
    envs = ps.difference(envs, EXCLUDE_ENVS)
    return envs


def gen_random_return(env_name, seed):
    '''Generate a single-episode random policy return for an environment'''
    # TODO generalize for unity too once it has a gym wrapper
    env = gym.make(env_name)
    env.seed(seed)
    env.reset()
    done = False
    total_reward = 0
    while not done:
        _, reward, done, _ = env.step(env.action_space.sample())
        total_reward += reward
    return total_reward


def gen_random_baseline(env_name, num_eval=NUM_EVAL):
    '''Generate the random baseline for an environment by averaging over num_eval episodes'''
    returns = util.parallelize(gen_random_return, [(env_name, i) for i in range(num_eval)])
    mean_rand_ret = np.mean(returns)
    std_rand_ret = np.std(returns)
    return {'mean': mean_rand_ret, 'std': std_rand_ret}


def main():
    '''
    Main method to generate all random baselines and write to file.
    Run as: python slm_lab/spec/random_baseline.py
    '''
    filepath = 'slm_lab/spec/_random_baseline.json'
    old_random_baseline = util.read(filepath)
    random_baseline = {}
    envs = enum_envs()
    for idx, env_name in enumerate(envs):
        if env_name in old_random_baseline:
            logger.info(f'Using existing random baseline for {env_name}: {idx + 1}/{len(envs)}')
            random_baseline[env_name] = old_random_baseline[env_name]
        else:
            try:
                logger.info(f'Generating random baseline for {env_name}: {idx + 1}/{len(envs)}')
                random_baseline[env_name] = gen_random_baseline(env_name, NUM_EVAL)
            except Exception as e:
                logger.warning(f'Cannot start env: {env_name}, skipping random baseline generation')
                continue
        util.write(random_baseline, filepath)
    logger.info(f'Done, random baseline written to {filepath}')
    return random_baseline


if __name__ == '__main__':
    main()
