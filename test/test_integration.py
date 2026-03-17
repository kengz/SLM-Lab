"""End-to-end integration tests — data flow only, no training.

Covers:
    1. Pavlovian env → DaseinNet forward → env.step (shape checks)
    2. Sensorimotor env → 56-dim obs → DaseinNet full pipeline → env.step
    3. EmotionModule → EmotionTag → EmotionTaggedReplayBuffer round-trip
    4. CurriculumSequencer task advancement via mock mastery
    5. run_eval with Pavlovian env + random agent
    6. run_eval with Sensorimotor env + random agent
    7. check_gate_min_pass pass/fail with mock EvalResults
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import torch

# Registration side-effect (must precede env imports)
import slm_lab.env  # noqa: F401

from slm_lab.agent.memory.emotion_replay import EmotionTaggedReplayBuffer, Transition
from slm_lab.agent.net.dasein_net import DaseinNet, OBS_DIM
from slm_lab.agent.net.emotion import EmotionModule, EmotionTag
from slm_lab.env.pavlovian import PavlovianEnv
from slm_lab.env.sensorimotor import SLMSensorimotor
from slm_lab.experiment.curriculum import CurriculumSequencer, MASTERY_WINDOW
from slm_lab.experiment.eval import EvalResults, run_eval
from slm_lab.experiment.gates import CHECKPOINT_A, check_gate_min_pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dasein_net(action_dim: int = 10) -> DaseinNet:
    net_spec = {
        "action_dim": action_dim,
        "log_std_init": 0.0,
        "clip_grad_val": 0.5,
        "optim_spec": {"name": "Adam", "lr": 3e-4},
        "gpu": False,
    }
    out_dim = [action_dim, action_dim, 1]
    return DaseinNet(net_spec=net_spec, in_dim=OBS_DIM, out_dim=out_dim)


class _RandomContinuousAgent:
    """Minimal agent with act(obs, deterministic) compatible with run_eval."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.action_space.sample()


class _ConstantScoreAgent:
    """Agent that always produces a fixed score via info['score']."""

    def __init__(self, action_space, score: float = 1.0):
        self.action_space = action_space
        self._score = score

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.action_space.sample()


def _make_eval_result(test_id: str, score: float, passed: bool) -> EvalResults:
    n = 20
    n_success = int(score * n)
    return EvalResults(
        test_id=test_id,
        n_trials=n,
        n_success=n_success,
        score=score,
        ci_lower=max(0.0, score - 0.05),
        ci_upper=min(1.0, score + 0.05),
        passed=passed,
    )


# ---------------------------------------------------------------------------
# 1. Pavlovian env → DaseinNet forward → env.step
# ---------------------------------------------------------------------------

class TestPavlovianDaseinForward:
    """DaseinNet accepts Pavlovian obs padded to 56-dim and produces valid action."""

    def test_pavlovian_dasein_forward(self):
        env = PavlovianEnv(task="stimulus_response", seed=0)
        net = _make_dasein_net(action_dim=10)
        net.eval()

        obs, _ = env.reset(seed=0)
        assert obs.shape == (18,), f"Expected (18,), got {obs.shape}"

        # Pad 18-dim Pavlovian obs to 56-dim for DaseinNet
        padded = np.zeros(OBS_DIM, dtype=np.float32)
        padded[:18] = obs

        x = torch.from_numpy(padded).unsqueeze(0)  # (1, 56)
        assert x.shape == (1, OBS_DIM)

        with torch.no_grad():
            out = net(x)

        mean, log_std, value = out
        assert mean.shape == (1, 10), f"mean shape: {mean.shape}"
        assert log_std.shape == (1, 10), f"log_std shape: {log_std.shape}"
        assert value.shape == (1, 1), f"value shape: {value.shape}"

        # Convert mean to numpy action (clipped to env action space)
        action = mean.squeeze(0).detach().numpy()[:2]  # Pavlovian uses 2-dim action
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        obs2, reward, terminated, truncated, info = env.step(action)

        assert obs2.shape == (18,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        env.close()


# ---------------------------------------------------------------------------
# 2. Sensorimotor env → 56-dim obs → DaseinNet full pipeline → env.step
# ---------------------------------------------------------------------------

class TestSensorimotorDaseinForward:
    """Full DaseinNet pipeline on sensorimotor 56-dim ground-truth obs."""

    def test_sensorimotor_dasein_forward(self):
        env = SLMSensorimotor(task_id="TC-13", seed=0)
        net = _make_dasein_net(action_dim=10)
        net.eval()

        obs_dict, _ = env.reset(seed=0)
        gt_obs = obs_dict["ground_truth"]

        assert gt_obs.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {gt_obs.shape}"

        x = torch.from_numpy(gt_obs).unsqueeze(0)  # (1, 56)

        # Verify obs split matches expected slices
        proprio = x[:, :25]
        tactile = x[:, 25:27]
        ee = x[:, 27:33]
        internal = x[:, 33:35]
        obj_state = x[:, 35:56]
        assert proprio.shape == (1, 25)
        assert tactile.shape == (1, 2)
        assert ee.shape == (1, 6)
        assert internal.shape == (1, 2)
        assert obj_state.shape == (1, 21)

        with torch.no_grad():
            out = net(x)

        mean, log_std, value = out
        assert mean.shape == (1, 10)
        assert log_std.shape == (1, 10)
        assert value.shape == (1, 1)

        # Action for env.step
        action = mean.squeeze(0).detach().numpy().astype(np.float32)
        action = np.clip(action, -1.0, 1.0)
        obs2_dict, reward, terminated, truncated, info = env.step(action)

        assert obs2_dict["ground_truth"].shape == (OBS_DIM,)
        assert isinstance(reward, float)

        env.close()


# ---------------------------------------------------------------------------
# 3. EmotionModule → EmotionTag → EmotionTaggedReplayBuffer
# ---------------------------------------------------------------------------

class TestEmotionReplayPipeline:
    """EmotionModule produces tags; tags drive priorities in replay buffer."""

    def test_emotion_replay_pipeline(self):
        module = EmotionModule(phase="3.2a")
        buf = EmotionTaggedReplayBuffer(capacity=1000, old_stage_reserve=0.10)

        env = PavlovianEnv(task="stimulus_response", seed=0)
        obs, _ = env.reset(seed=0)

        n_transitions = 50
        for i in range(n_transitions):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)

            pe = float(np.abs(reward))  # proxy prediction error
            tag: EmotionTag = module.compute(pe=pe, reward=reward)

            assert tag.emotion_type in (
                "fear", "surprise", "satisfaction", "frustration",
                "curiosity", "social_approval", "neutral",
            )
            assert 0.0 <= tag.magnitude <= 1.0

            transition = Transition(
                state=obs.astype(np.float32),
                action=action.astype(np.float32),
                reward=float(reward),
                next_state=next_obs.astype(np.float32),
                done=terminated or truncated,
                emotion_type=tag.emotion_type,
                emotion_magnitude=tag.magnitude,
                prediction_error=pe,
                stage_name="pavlovian",
            )
            buf.add(transition)

            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

        assert buf.size == n_transitions

        # Sample a batch and verify shapes
        batch_size = 16
        transitions, is_weights = buf.sample_batch(batch_size)

        assert len(transitions) > 0
        assert len(is_weights) == len(transitions)
        assert is_weights.dtype == np.float32

        states = np.stack([t.state for t in transitions])
        assert states.shape == (len(transitions), 18)

        env.close()


# ---------------------------------------------------------------------------
# 4. CurriculumSequencer task advancement via mock mastery
# ---------------------------------------------------------------------------

class TestCurriculumProgression:
    """Sequencer advances when mastery window is met."""

    def test_curriculum_progression(self):
        seq = CurriculumSequencer(
            max_attempts_per_task=10000,
            mastery_threshold=0.80,
            mastery_window=MASTERY_WINDOW,
        )

        first_task = seq.current_task
        assert first_task == "stimulus_response"

        # Feed enough high scores to trigger mastery
        for _ in range(MASTERY_WINDOW):
            seq.record_episode(first_task, score=1.0)

        advanced = seq.advance_if_ready()
        assert advanced is True, "Expected advancement after mastery"

        second_task = seq.current_task
        assert second_task != first_task
        assert second_task == "temporal_contingency"

    def test_curriculum_stuck_advancement(self):
        seq = CurriculumSequencer(
            max_attempts_per_task=5,
            mastery_threshold=0.80,
            mastery_window=MASTERY_WINDOW,
        )

        task = seq.current_task
        # Feed poor scores; should advance after max_attempts
        for _ in range(5):
            seq.record_episode(task, score=0.0)

        advanced = seq.advance_if_ready()
        assert advanced is True
        assert seq.state.task_records[task].flagged_stuck is True


# ---------------------------------------------------------------------------
# 5. run_eval with Pavlovian env + random agent
# ---------------------------------------------------------------------------

class TestEvalWithPavlovian:
    """run_eval completes and returns valid EvalResults for Pavlovian env."""

    def test_eval_with_pavlovian(self):
        env = PavlovianEnv(task="stimulus_response", seed=0)
        agent = _RandomContinuousAgent(env.action_space)

        results = run_eval(
            env=env,
            agent=agent,
            n_trials=3,
            score_type="binary",
            test_id="TC-01-pavlovian-random",
            threshold=0.0,
        )

        assert isinstance(results, EvalResults)
        assert results.test_id == "TC-01-pavlovian-random"
        assert results.n_trials == 3
        assert 0.0 <= results.score <= 1.0
        assert 0.0 <= results.ci_lower <= results.ci_upper <= 1.0
        assert isinstance(results.passed, bool)

        env.close()


# ---------------------------------------------------------------------------
# 6. run_eval with Sensorimotor env + random agent
# ---------------------------------------------------------------------------

class TestEvalWithSensorimotor:
    """run_eval completes and returns valid EvalResults for Sensorimotor env."""

    def test_eval_with_sensorimotor(self):
        env = SLMSensorimotor(task_id="TC-11", seed=0)

        # Sensorimotor obs is a dict; need wrapper for run_eval
        class _SensorimotorAgent:
            def __init__(self, action_space):
                self.action_space = action_space

            def act(self, obs, deterministic: bool = True):
                # obs is dict with "ground_truth" key
                return self.action_space.sample()

        agent = _SensorimotorAgent(env.action_space)

        results = run_eval(
            env=env,
            agent=agent,
            n_trials=2,
            score_type="binary",
            test_id="TC-11-sensorimotor-random",
            threshold=0.0,
        )

        assert isinstance(results, EvalResults)
        assert results.n_trials == 2
        assert 0.0 <= results.score <= 1.0
        assert isinstance(results.passed, bool)

        env.close()


# ---------------------------------------------------------------------------
# 7. check_gate_min_pass pass/fail
# ---------------------------------------------------------------------------

class TestGateCheckpointA:
    """check_gate_min_pass correctly enforces ≥6/10 criterion."""

    def _build_results(self, passing_tasks: list[str], failing_tasks: list[str]) -> dict[str, EvalResults]:
        results: dict[str, EvalResults] = {}
        for task in passing_tasks:
            threshold = CHECKPOINT_A.criteria[task]
            results[task] = _make_eval_result(task, score=threshold + 0.01, passed=True)
        for task in failing_tasks:
            threshold = CHECKPOINT_A.criteria[task]
            results[task] = _make_eval_result(task, score=max(0.0, threshold - 0.1), passed=False)
        return results

    def test_gate_passes_with_six_tasks(self):
        passing = list(CHECKPOINT_A.criteria.keys())[:6]
        failing = list(CHECKPOINT_A.criteria.keys())[6:]
        results = self._build_results(passing, failing)

        gr = check_gate_min_pass(results, CHECKPOINT_A, min_passing=6)
        assert gr.passed is True
        assert len(gr.passing) == 6

    def test_gate_fails_with_five_tasks(self):
        passing = list(CHECKPOINT_A.criteria.keys())[:5]
        failing = list(CHECKPOINT_A.criteria.keys())[5:]
        results = self._build_results(passing, failing)

        gr = check_gate_min_pass(results, CHECKPOINT_A, min_passing=6)
        assert gr.passed is False
        assert len(gr.passing) == 5

    def test_gate_passes_with_all_tasks(self):
        passing = list(CHECKPOINT_A.criteria.keys())
        results = self._build_results(passing, [])

        gr = check_gate_min_pass(results, CHECKPOINT_A, min_passing=6)
        assert gr.passed is True
        assert len(gr.failing) == 0
        assert len(gr.missing) == 0
