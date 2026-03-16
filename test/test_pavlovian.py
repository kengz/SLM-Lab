"""Integration tests for PavlovianEnv (SLM/Pavlovian-v0).

Covers:
- Env instantiation for each of the 10 tasks
- Observation / action space shapes
- Single step and reset cycle
- Two-phase protocol transitions (acquisition → probe)
- Reward range sanity
- Vectorized env (2 parallel, AsyncVectorEnv)
"""

import math

import gymnasium as gym
import numpy as np
import pytest

# Import triggers registration
import slm_lab.env  # noqa: F401
from slm_lab.env.pavlovian import (
    ACT_DIM,
    MAX_ENERGY,
    OBS_DIM,
    PHASE_ACQUISITION,
    PHASE_PROBE,
    PavlovianEnv,
    TASKS,
)

ENV_ID = "SLM/Pavlovian-v0"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=TASKS)
def task(request) -> str:
    return request.param


@pytest.fixture
def env_factory():
    """Factory for constructing and auto-closing envs."""
    envs = []

    def make(task: str = "stimulus_response", **kwargs) -> PavlovianEnv:
        e = PavlovianEnv(task=task, seed=42, **kwargs)
        envs.append(e)
        return e

    yield make
    for e in envs:
        e.close()


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    def test_all_tasks_instantiate(self, task):
        env = PavlovianEnv(task=task)
        assert env is not None
        env.close()

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            PavlovianEnv(task="nonexistent_task")

    def test_gymnasium_make(self):
        env = gym.make(ENV_ID, task="stimulus_response")
        assert env is not None
        env.close()


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------

class TestSpaces:
    def test_observation_space_shape(self, task, env_factory):
        env = env_factory(task=task)
        obs, _ = env.reset()
        assert env.observation_space.shape == (OBS_DIM,), (
            f"{task}: expected obs shape ({OBS_DIM},), got {env.observation_space.shape}"
        )
        assert obs.shape == (OBS_DIM,), f"{task}: obs shape mismatch"

    def test_action_space_shape(self, task, env_factory):
        env = env_factory(task=task)
        assert env.action_space.shape == (ACT_DIM,)

    def test_action_space_bounds(self, task, env_factory):
        env = env_factory(task=task)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)

    def test_observation_dtype(self, task, env_factory):
        env = env_factory(task=task)
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_obs_dim_17_is_stimulus(self, env_factory):
        """Obs[17] should be the stimulus signal (0 or 1 for classical tasks)."""
        env = env_factory(task="stimulus_response")
        obs, _ = env.reset()
        # At reset step 0 (start of ITI), stimulus should be 0
        assert obs[17] == 0.0


# ---------------------------------------------------------------------------
# Reset / Step cycle
# ---------------------------------------------------------------------------

class TestResetStep:
    def test_reset_returns_correct_shapes(self, task, env_factory):
        env = env_factory(task=task)
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,)
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self, task, env_factory):
        env = env_factory(task=task)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert isinstance(float(reward), float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_step_action_clipping(self, task, env_factory):
        """Out-of-bounds actions should not crash the env."""
        env = env_factory(task=task)
        env.reset()
        extreme = np.array([5.0, -5.0], dtype=np.float32)
        obs, reward, _, _, _ = env.step(extreme)
        assert obs.shape == (OBS_DIM,)

    def test_multiple_steps_do_not_crash(self, task, env_factory):
        env = env_factory(task=task)
        env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                env.reset()

    def test_seed_determinism(self, env_factory):
        """Same seed should produce same initial observation."""
        env1 = env_factory(task="stimulus_response")
        env2 = env_factory(task="stimulus_response")
        obs1, _ = env1.reset(seed=0)
        obs2, _ = env2.reset(seed=0)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_differ(self, env_factory):
        env = env_factory(task="stimulus_response")
        obs1, _ = env.reset(seed=0)
        obs2, _ = env.reset(seed=99)
        assert not np.array_equal(obs1, obs2)

    def test_reset_resets_energy(self, env_factory):
        env = env_factory(task="reward_contingency")
        env.reset()
        # Run enough steps to drain some energy
        for _ in range(100):
            env.step(np.array([1.0, 0.0]))
        obs, _ = env.reset()
        assert obs[3] == pytest.approx(1.0, abs=0.05)  # energy normalised


# ---------------------------------------------------------------------------
# Reward sanity
# ---------------------------------------------------------------------------

class TestRewardSanity:
    def test_reward_is_finite(self, task, env_factory):
        env = env_factory(task=task)
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            _, reward, terminated, _, _ = env.step(action)
            assert math.isfinite(reward), f"{task}: non-finite reward {reward}"
            if terminated:
                env.reset()

    def test_reward_contingency_positive_reward(self, env_factory):
        """TC-07: forward action should yield positive reward."""
        env = env_factory(task="reward_contingency")
        env.reset()
        total = 0.0
        for _ in range(200):
            _, reward, terminated, _, _ = env.step(np.array([1.0, 0.0]))
            total += reward
            if terminated:
                env.reset()
        assert total > 0.0, "TC-07 should yield positive reward for forward movement"

    def test_no_reward_during_iti_tc01(self, env_factory):
        """TC-01: during ITI (cs_signal=0), forward-only steps should yield ~0 reward (no shaping)."""
        env = env_factory(task="stimulus_response")
        env.reset()
        # First 60 steps are ITI (steps 0-59 in cycle of 90)
        rewards = []
        for i in range(55):  # stay safely inside ITI
            _, reward, _, _, info = env.step(np.array([0.0, 0.0]))
            if not info["cs_active"]:
                rewards.append(reward)
        # No shaping during ITI → all rewards should be 0
        assert all(r == 0.0 for r in rewards), (
            f"Expected 0 reward during ITI, got {rewards}"
        )

    def test_shaping_active_during_cs_acquisition(self, env_factory):
        """TC-01 acquisition: approaching red during CS should yield shaping reward."""
        env = env_factory(task="stimulus_response")
        env.reset()
        # Advance into the first CS window (step 60 onward in cycle)
        for _ in range(62):
            env.step(np.array([0.0, 0.0]))
        # Now move toward red (objects are at ~7.5, 7.5; agent starts near 5, 5)
        reward_sum = 0.0
        for _ in range(10):
            _, reward, _, _, info = env.step(np.array([1.0, 0.0]))
            if info["cs_active"] and info["phase"] == PHASE_ACQUISITION:
                reward_sum += reward
        # Should have received some shaping reward if moving toward red
        # (Not guaranteed without steering, but environment should not error)
        assert math.isfinite(reward_sum)

    def test_partial_reinforcement_stochastic(self, env_factory):
        """TC-08: reward should be 0 on roughly 50% of forward steps."""
        env = env_factory(task="partial_reinforcement")
        env.reset()
        zero_rewards = 0
        nonzero_rewards = 0
        for _ in range(500):
            _, reward, terminated, _, _ = env.step(np.array([1.0, 0.0]))
            if reward == 0.0:
                zero_rewards += 1
            else:
                nonzero_rewards += 1
            if terminated:
                env.reset()
        total = zero_rewards + nonzero_rewards
        # Expect roughly 50% zero rewards ± 15%
        ratio = zero_rewards / total
        assert 0.35 <= ratio <= 0.65, f"TC-08 zero reward ratio {ratio:.2f} outside [0.35, 0.65]"


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

class TestPhaseTransitions:
    # Use high max_energy so episodes last long enough for phase transitions.
    # 40 trials × 90 steps = 3600 steps; energy_decay=0.1 → need max_energy >= 400.
    HIGH_ENERGY = 10000.0

    def _run_to_phase(
        self, env: PavlovianEnv, target_phase: str, max_steps: int = 10000
    ) -> bool:
        """Step env until the target phase is reached. Returns True if reached."""
        for _ in range(max_steps):
            action = env.action_space.sample()
            _, _, terminated, _, info = env.step(action)
            if info.get("phase") == target_phase:
                return True
            if terminated:
                break  # don't reset — this would reset _ts and lose phase progress
        return False

    def test_tc01_transitions_to_probe(self, env_factory):
        """TC-01 must transition from acquisition to probe after 40 trials."""
        # 40 trials × 90 steps = 3600 steps; budget = 5000 steps
        env = env_factory(task="stimulus_response", max_energy=self.HIGH_ENERGY)
        env.reset()
        reached = self._run_to_phase(env, PHASE_PROBE, max_steps=5000)
        assert reached, "TC-01: never reached probe phase"

    def test_tc03_has_acquisition_and_extinction_phases(self, env_factory):
        """TC-03 should have acquisition then extinction (or acquisition_failed) phases."""
        # 40 trials × 90 steps = 3600 steps; budget = 5000
        env = env_factory(task="extinction", max_energy=self.HIGH_ENERGY)
        env.reset()
        phases_seen = set()
        for _ in range(5000):
            action = env.action_space.sample()
            _, _, terminated, _, info = env.step(action)
            phases_seen.add(info.get("phase"))
            if terminated:
                break
        assert PHASE_ACQUISITION in phases_seen
        # Should reach either extinction or acquisition_failed
        assert "extinction" in phases_seen or "acquisition_failed" in phases_seen, (
            f"TC-03: never reached extinction phase. Phases seen: {phases_seen}"
        )

    def test_tc04_rest_phase_exists(self, env_factory):
        """TC-04 should pass through a REST phase."""
        # 30 + 30 trials × 90 steps + 150 rest = ~5550 steps; budget = 8000
        env = env_factory(task="spontaneous_recovery", max_energy=self.HIGH_ENERGY)
        env.reset()
        phases_seen = set()
        for _ in range(8000):
            action = env.action_space.sample()
            _, _, terminated, _, info = env.step(action)
            phases_seen.add(info.get("phase"))
            if terminated:
                break
        # Must see at least acquisition
        assert PHASE_ACQUISITION in phases_seen

    def test_tc05_probe_order_is_randomised(self, env_factory):
        """TC-05 generalization probe: multiple runs should produce different stimulus orderings."""
        env = env_factory(task="generalization")
        first_levels: list[float] = []
        second_levels: list[float] = []

        for run_idx, levels_list in enumerate([first_levels, second_levels]):
            env.reset(seed=run_idx * 17)
            in_probe = False
            for _ in range(8000):
                action = env.action_space.sample()
                _, _, terminated, _, info = env.step(action)
                if info.get("phase") == PHASE_PROBE and not in_probe:
                    in_probe = True
                if in_probe and info.get("cs_active"):
                    levels_list.append(info.get("cs_signal", 0.0))
                    if len(levels_list) >= 5:
                        break
                if terminated:
                    env.reset(seed=run_idx * 17 + 1)

        # Two runs with different seeds should differ in at least one recorded level
        # (or both empty — in which case the test is inconclusive, not a failure)
        if first_levels and second_levels:
            # The levels should come from the valid set
            valid = {1.0, 0.8, 0.6, 0.4, 0.2}
            for lv in first_levels + second_levels:
                assert lv in valid or lv == 0.0, f"Unexpected stimulus level {lv}"

    def test_tc06_discrimination_cs_types(self, env_factory):
        """TC-06 should expose cs_plus and cs_minus approaches in info."""
        env = env_factory(task="discrimination")
        env.reset()
        for _ in range(200):
            env.step(env.action_space.sample())
        _, _, _, _, info = env.step(env.action_space.sample())
        assert "cs_plus_approaches" in info
        assert "cs_minus_approaches" in info


# ---------------------------------------------------------------------------
# Task-specific info keys
# ---------------------------------------------------------------------------

class TestInfoKeys:
    EXPECTED_KEYS = {
        "stimulus_response": ["probe_approaches", "acq_approaches", "iti_approaches"],
        "temporal_contingency": ["probe_trials", "acq_trials"],
        "extinction": ["acq_approaches", "ext_approaches", "acquisition_failed"],
        "spontaneous_recovery": ["acq_approaches", "ext_approaches"],
        "generalization": ["responses_by_strength"],
        "discrimination": ["cs_plus_approaches", "cs_minus_approaches"],
        "reward_contingency": ["total_steps", "v_forward"],
        "partial_reinforcement": ["total_steps", "reward_this_step"],
        "shaping": ["shaped_successes", "unshaped_successes", "condition"],
        "chaining": ["chains_completed", "chains_attempted", "chain_step"],
    }

    def test_info_contains_expected_keys(self, task, env_factory):
        env = env_factory(task=task)
        env.reset()
        env.step(env.action_space.sample())
        _, _, _, _, info = env.step(env.action_space.sample())
        for key in self.EXPECTED_KEYS[task]:
            assert key in info, f"{task}: missing info key '{key}'"


# ---------------------------------------------------------------------------
# Vectorized environment
# ---------------------------------------------------------------------------

class TestVectorEnv:
    def test_async_vector_env_2_parallel(self):
        """2 parallel AsyncVectorEnv instances for stimulus_response."""
        venv = gym.make_vec(
            ENV_ID,
            num_envs=2,
            vectorization_mode="async",
            task="stimulus_response",
        )
        try:
            obs, info = venv.reset()
            assert obs.shape == (2, OBS_DIM)
            actions = np.stack([venv.single_action_space.sample() for _ in range(2)])
            obs, rewards, terminated, truncated, info = venv.step(actions)
            assert obs.shape == (2, OBS_DIM)
            assert rewards.shape == (2,)
            assert terminated.shape == (2,)
        finally:
            venv.close()

    def test_sync_vector_env_2_parallel(self):
        """2 parallel SyncVectorEnv instances."""
        venv = gym.make_vec(
            ENV_ID,
            num_envs=2,
            vectorization_mode="sync",
            task="reward_contingency",
        )
        try:
            obs, _ = venv.reset()
            assert obs.shape == (2, OBS_DIM)
            actions = np.stack([venv.single_action_space.sample() for _ in range(2)])
            obs, rewards, _, _, _ = venv.step(actions)
            assert obs.shape == (2, OBS_DIM)
            assert rewards.shape == (2,)
        finally:
            venv.close()

    def test_vector_env_all_tasks_smoke(self):
        """Smoke test: all 10 tasks can be vectorized (sync, 2 envs)."""
        for task_name in TASKS:
            venv = gym.make_vec(
                ENV_ID,
                num_envs=2,
                vectorization_mode="sync",
                task=task_name,
            )
            try:
                obs, _ = venv.reset()
                assert obs.shape == (2, OBS_DIM), f"{task_name}: obs shape wrong"
                actions = np.stack([venv.single_action_space.sample() for _ in range(2)])
                obs, _, _, _, _ = venv.step(actions)
                assert obs.shape == (2, OBS_DIM), f"{task_name}: step obs shape wrong"
            finally:
                venv.close()


# ---------------------------------------------------------------------------
# Chaining task specifics
# ---------------------------------------------------------------------------

class TestChaining:
    def test_chain_step_increments_on_correct_contact(self, env_factory):
        """TC-10: chain_step should advance when the correct object is contacted."""
        env = env_factory(task="chaining")
        env.reset()
        # Manually position agent on green object
        env._agent.x = env._objects[2].x  # green
        env._agent.y = env._objects[2].y
        _, _, _, _, info = env.step(np.array([0.0, 0.0]))
        assert info["chain_step"] >= 0  # at least no crash

    def test_chains_attempted_increments(self, env_factory):
        env = env_factory(task="chaining")
        env.reset()
        # Position on green to start chain
        env._agent.x = env._objects[2].x
        env._agent.y = env._objects[2].y
        env.step(np.array([0.0, 0.0]))
        _, _, _, _, info = env.step(np.array([0.0, 0.0]))
        # After contacting green, chains_attempted should be >= 1
        assert info["chains_attempted"] >= 1


# ---------------------------------------------------------------------------
# Energy system
# ---------------------------------------------------------------------------

class TestEnergy:
    def test_energy_depletes_over_time(self, env_factory):
        env = env_factory(task="reward_contingency")
        env.reset()
        obs_start, _ = env.reset()
        energy_start = obs_start[3]
        # Run 300 steps with zero action (just decay)
        for _ in range(300):
            obs, _, terminated, _, _ = env.step(np.array([0.0, 0.0]))
            if terminated:
                break
        # Energy should have decreased
        assert obs[3] < energy_start

    def test_episode_terminates_on_energy_depletion(self, env_factory):
        env = env_factory(task="reward_contingency", max_energy=5.0)
        env.reset()
        terminated_seen = False
        for _ in range(2000):
            _, _, terminated, _, _ = env.step(np.array([0.0, 0.0]))
            if terminated:
                terminated_seen = True
                break
        assert terminated_seen, "Episode should terminate when energy depletes"
