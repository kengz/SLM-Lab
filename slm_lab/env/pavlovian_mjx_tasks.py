"""Pure JAX reward functions for Pavlovian TC-01 through TC-10.

Each task function has signature:
    (data, action, info, metrics) -> (reward, info, metrics)

IMPORTANT: task functions must NOT add new keys to metrics — JAX scan
requires fixed pytree structure. All metric keys must be pre-initialized
in TASK_METRIC_KEYS and populated in reset().
"""
import jax.numpy as jp


def _not_implemented(data, action, info, metrics):
    """Placeholder for unimplemented tasks."""
    return jp.float32(0.0), info, metrics


def reward_tc07(data, action, info, metrics):
    """TC-07: Reward contingency — forward velocity reward."""
    fwd = jp.clip(action[0], 0.0, 1.0)
    reward = fwd * 0.5
    energy = info["energy"] + reward * 0.1
    info = {**info, "energy": energy}
    metrics["reward/forward"] = reward
    return reward, info, metrics


# Task function table — indexed by task_id - 1
TASK_FNS = [
    _not_implemented,  # TC-01
    _not_implemented,  # TC-02
    _not_implemented,  # TC-03
    _not_implemented,  # TC-04
    _not_implemented,  # TC-05
    _not_implemented,  # TC-06
    reward_tc07,       # TC-07
    _not_implemented,  # TC-08
    _not_implemented,  # TC-09
    _not_implemented,  # TC-10
]

# Extra metric keys per task (beyond "reward/task" which is always present).
# Must match exactly the keys written by the corresponding reward function.
TASK_METRIC_KEYS: list[list[str]] = [
    [],                   # TC-01
    [],                   # TC-02
    [],                   # TC-03
    [],                   # TC-04
    [],                   # TC-05
    [],                   # TC-06
    ["reward/forward"],   # TC-07
    [],                   # TC-08
    [],                   # TC-09
    [],                   # TC-10
]
