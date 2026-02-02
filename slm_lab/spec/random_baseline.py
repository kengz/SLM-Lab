# Module to generate random baselines
# Run as: python slm_lab/spec/random_baseline.py
from slm_lab.lib import logger, util
import gymnasium as gym
import numpy as np

# Ensure ALE environments are registered (same as slm_lab.env.gym)
try:
    import ale_py
    import os
    import warnings
    # Silence ALE output more aggressively
    os.environ['ALE_PY_SILENCE'] = '1'
    warnings.filterwarnings('ignore', category=UserWarning, module='ale_py')
    gym.register_envs(ale_py)
except ImportError:
    pass


FILEPATH = 'slm_lab/spec/_random_baseline.json'
NUM_EVAL = 100


def enum_envs():
    '''Enumerate only the latest version of each environment, preferring ALE/ over legacy variants'''
    envs = []
    # Skip problematic environments that fail during random baseline generation
    skip_envs = {
        'tabular/Blackjack-v0', 
        'tabular/CliffWalking-v0',
        'GymV21Environment-v0',
        'GymV26Environment-v0',
        'phys2d/CartPole-v0',
        'phys2d/CartPole-v1',
        'phys2d/Pendulum-v0'
    }
    
    for env_spec in gym.envs.registry.values():
        env_id = env_spec.id
        entry_point = str(env_spec.entry_point).lower()
        
        # Skip known problematic environments
        if env_id in skip_envs:
            continue
        
        # For ALE environments, only keep ALE/ prefixed ones
        if 'ale_py' in entry_point and not env_id.startswith('ALE/'):
            continue  # Skip legacy atari variants
            
        envs.append(env_id)
    
    # Get latest version of each environment family
    envs = sorted(envs)
    latest = {}
    for env_id in envs:
        base = env_id.split('-v')[0] if '-v' in env_id else env_id
        latest[base] = env_id  # sorted order ensures latest overwrites earlier
    return list(latest.values())


def gen_random_return(env_name, seed):
    '''Generate a single-episode random policy return for an environment'''
    env = gym.make(env_name)  # No render_mode = no rendering (headless)
    state, info = env.reset(seed=seed)
    
    total_reward = 0
    while True:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    env.close()
    return total_reward


def gen_random_baseline(env_name, num_eval=NUM_EVAL):
    '''Generate the random baseline for an environment by averaging over num_eval episodes'''
    returns = util.parallelize(gen_random_return, [(env_name, i) for i in range(num_eval)])
    mean_rand_ret = np.mean(returns)
    std_rand_ret = np.std(returns)
    return {'mean': mean_rand_ret, 'std': std_rand_ret}


def get_random_baseline(env_name):
    '''Get a single random baseline for env; if does not exist in file, generate live and update the file'''
    random_baseline = util.read(FILEPATH)
    if env_name in random_baseline:
        baseline = random_baseline[env_name]
    else:
        try:
            logger.info(f'Generating random baseline for {env_name}')
            baseline = gen_random_baseline(env_name, NUM_EVAL)
        except Exception:
            logger.warning(f'Cannot start env: {env_name}, skipping random baseline generation')
            baseline = None
        # update immediately
        logger.info(f'Updating new random baseline in {FILEPATH}')
        random_baseline[env_name] = baseline
        util.write(random_baseline, FILEPATH)
    return baseline


def main():
    '''
    Main method to generate all random baselines and write to file.
    Run as: python slm_lab/spec/random_baseline.py
    '''
    envs = enum_envs()
    logger.info(f'Will generate random baselines for {len(envs)} environments:')
    for env_name in envs:
        logger.info(f'  - {env_name}')
    logger.info('')
    
    for idx, env_name in enumerate(envs):
        logger.info(f'Generating random baseline for {env_name}: {idx + 1}/{len(envs)}')
        get_random_baseline(env_name)
    logger.info(f'Done, random baseline updated in {FILEPATH}')


if __name__ == '__main__':
    main()
