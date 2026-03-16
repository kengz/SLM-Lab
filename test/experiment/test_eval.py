# Unit tests for eval.py and gates.py
import math

import numpy as np
import pytest

from slm_lab.experiment.eval import (
    EvalResults,
    bootstrap_ci,
    check_threshold,
    clopper_pearson_ci,
    compute_ci,
    format_results,
    iqm,
    run_eval,
)
from slm_lab.experiment.gates import (
    CHECKPOINT_A,
    CHECKPOINT_B,
    CHECKPOINT_D,
    DINO_PROBE_GATE,
    GateConfig,
    GateResult,
    check_gate,
    check_gate_min_pass,
)


# ---------------------------------------------------------------------------
# Helpers / Stubs
# ---------------------------------------------------------------------------

class _ConstEnv:
    """Minimal env stub: each episode returns one step then terminates.
    info always contains {"score": score, "is_success": True/False}.
    """
    def __init__(self, score: float = 1.0):
        self._score = score

    def reset(self, seed: int | None = None):
        return np.zeros(4), {}

    def step(self, action):
        info = {"score": self._score, "is_success": self._score >= 0.5}
        return np.zeros(4), 0.0, True, False, info


class _SuccessKeyEnv:
    """Uses is_success instead of score key."""
    def reset(self, seed=None):
        return np.zeros(4), {}

    def step(self, action):
        info = {"is_success": True}
        return np.zeros(4), 0.0, True, False, info


class _StubAgent:
    def act(self, obs, deterministic: bool = True):
        return np.zeros(2)


# ---------------------------------------------------------------------------
# clopper_pearson_ci
# ---------------------------------------------------------------------------

class TestClopperPearsonCI:
    def test_all_success(self):
        lo, hi = clopper_pearson_ci(10, 10)
        assert lo > 0.69
        assert hi == pytest.approx(1.0, abs=1e-6)

    def test_no_success(self):
        lo, hi = clopper_pearson_ci(0, 10)
        assert lo == pytest.approx(0.0, abs=1e-6)
        assert hi < 0.31

    def test_half_success(self):
        lo, hi = clopper_pearson_ci(5, 10)
        assert lo < 0.5 < hi

    def test_zero_trials(self):
        lo, hi = clopper_pearson_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_asymmetry(self):
        lo, hi = clopper_pearson_ci(1, 10)
        assert hi - 0.1 > 0.1 - lo  # CI is wider on upper side near 0


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:
    def test_all_ones(self):
        lo, hi = bootstrap_ci([1.0] * 20)
        assert lo == pytest.approx(1.0, abs=1e-6)
        assert hi == pytest.approx(1.0, abs=1e-6)

    def test_all_zeros(self):
        lo, hi = bootstrap_ci([0.0] * 20)
        assert lo == pytest.approx(0.0, abs=1e-6)

    def test_mixed(self):
        scores = [0.0] * 10 + [1.0] * 10
        lo, hi = bootstrap_ci(scores, seed=0)
        assert 0.0 < lo < 0.5 < hi < 1.0

    def test_ci_width_decreases_with_n(self):
        rng = np.random.default_rng(7)
        small = rng.random(10).tolist()
        large = rng.random(100).tolist()
        lo_s, hi_s = bootstrap_ci(small, seed=0)
        lo_l, hi_l = bootstrap_ci(large, seed=0)
        assert (hi_s - lo_s) > (hi_l - lo_l)


# ---------------------------------------------------------------------------
# compute_ci
# ---------------------------------------------------------------------------

class TestComputeCI:
    def test_binary_delegates_to_clopper_pearson(self):
        scores = [1.0] * 8 + [0.0] * 2
        lo, hi = compute_ci(scores, score_type="binary")
        lo_ref, hi_ref = clopper_pearson_ci(8, 10)
        assert lo == pytest.approx(lo_ref, abs=1e-6)
        assert hi == pytest.approx(hi_ref, abs=1e-6)

    def test_continuous_delegates_to_bootstrap(self):
        scores = [0.3, 0.5, 0.7, 0.9, 0.4]
        lo, hi = compute_ci(scores, score_type="continuous")
        assert 0.0 < lo < hi < 1.0


# ---------------------------------------------------------------------------
# check_threshold
# ---------------------------------------------------------------------------

class TestCheckThreshold:
    def _make_results(self, score: float, ci_lower: float) -> EvalResults:
        return EvalResults(
            test_id="TC-00", n_trials=10, n_success=5,
            score=score, ci_lower=ci_lower, ci_upper=1.0, passed=False,
        )

    def test_pass_no_ci_threshold(self):
        r = self._make_results(0.85, 0.60)
        assert check_threshold(r, threshold=0.80)

    def test_fail_score_below_threshold(self):
        r = self._make_results(0.75, 0.60)
        assert not check_threshold(r, threshold=0.80)

    def test_pass_with_ci_threshold(self):
        r = self._make_results(0.85, 0.58)
        assert check_threshold(r, threshold=0.80, ci_threshold=0.56)

    def test_fail_ci_below_ci_threshold(self):
        r = self._make_results(0.85, 0.50)
        assert not check_threshold(r, threshold=0.80, ci_threshold=0.56)


# ---------------------------------------------------------------------------
# iqm
# ---------------------------------------------------------------------------

class TestIQM:
    def test_simple(self):
        scores = [0.1, 0.2, 0.5, 0.8, 0.9]
        result = iqm(scores)
        # middle 50%: indices 1..3 → [0.2, 0.5, 0.8]
        assert result == pytest.approx(np.mean([0.2, 0.5, 0.8]), abs=1e-6)

    def test_all_same(self):
        assert iqm([0.7] * 10) == pytest.approx(0.7, abs=1e-6)

    def test_single_element(self):
        assert iqm([0.5]) == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------

class TestRunEval:
    def test_all_success(self):
        env = _ConstEnv(score=1.0)
        agent = _StubAgent()
        results = run_eval(env, agent, n_trials=10, test_id="TC-01", threshold=0.80)
        assert results.score == pytest.approx(1.0)
        assert results.n_success == 10
        assert results.passed is True
        assert results.ci_lower > 0.69

    def test_all_fail(self):
        env = _ConstEnv(score=0.0)
        agent = _StubAgent()
        results = run_eval(env, agent, n_trials=10, test_id="TC-01", threshold=0.80)
        assert results.score == pytest.approx(0.0)
        assert results.passed is False

    def test_n_trials_respected(self):
        env = _ConstEnv(score=0.9)
        agent = _StubAgent()
        results = run_eval(env, agent, n_trials=20)
        assert results.n_trials == 20
        assert len(results.trial_scores) == 20

    def test_is_success_fallback(self):
        env = _SuccessKeyEnv()
        agent = _StubAgent()
        results = run_eval(env, agent, n_trials=5, threshold=0.80)
        assert results.score == pytest.approx(1.0)

    def test_threshold_and_ci_threshold(self):
        # 7/10 successes: score=0.7, CI lower ~0.35 — fails ci_threshold=0.56
        env = _ConstEnv(score=0.7)  # score per trial is 0.7, so all "success" (>=0.5)
        agent = _StubAgent()
        results = run_eval(
            env, agent, n_trials=10, threshold=0.65, ci_threshold=0.56
        )
        # score = 0.7 (mean), CI on 10/10 binary successes > 0.56
        assert results.score == pytest.approx(0.7)

    def test_metrics_aggregated(self):
        class _MetricEnv:
            def reset(self, seed=None):
                return np.zeros(2), {}
            def step(self, a):
                return np.zeros(2), 0.0, True, False, {"score": 1.0, "approach_rate": 0.9}

        results = run_eval(_MetricEnv(), _StubAgent(), n_trials=3)
        assert "approach_rate" in results.metrics
        assert results.metrics["approach_rate"] == pytest.approx(0.9)

    def test_format_results(self):
        env = _ConstEnv(score=1.0)
        agent = _StubAgent()
        results = run_eval(env, agent, n_trials=5, test_id="TC-01")
        text = format_results(results)
        assert "TC-01" in text
        assert "Pass" in text
        assert "CI" in text


# ---------------------------------------------------------------------------
# check_gate
# ---------------------------------------------------------------------------

class TestCheckGate:
    def _make_results(self, score: float, test_id: str = "task_a") -> EvalResults:
        return EvalResults(
            test_id=test_id, n_trials=10, n_success=int(score * 10),
            score=score, ci_lower=score - 0.1, ci_upper=score + 0.1, passed=score >= 0.5,
        )

    def test_all_pass(self):
        gate = GateConfig(name="TEST", criteria={"a": 0.80, "b": 0.60})
        results = {
            "a": self._make_results(0.85, "a"),
            "b": self._make_results(0.70, "b"),
        }
        gr = check_gate(results, gate)
        assert gr.passed is True
        assert gr.failing == {}
        assert gr.missing == []

    def test_one_fails(self):
        gate = GateConfig(name="TEST", criteria={"a": 0.80, "b": 0.60})
        results = {
            "a": self._make_results(0.75, "a"),  # below 0.80
            "b": self._make_results(0.70, "b"),
        }
        gr = check_gate(results, gate)
        assert gr.passed is False
        assert "a" in gr.failing

    def test_missing_task_fails_gate(self):
        gate = GateConfig(name="TEST", criteria={"a": 0.80, "b": 0.60})
        results = {"a": self._make_results(0.90, "a")}
        gr = check_gate(results, gate)
        assert gr.passed is False
        assert "b" in gr.missing

    def test_empty_criteria_passes(self):
        gate = GateConfig(name="EMPTY", criteria={})
        gr = check_gate({}, gate)
        assert gr.passed is True

    def test_gate_result_summary(self):
        gate = GateConfig(name="TEST", criteria={"a": 0.80})
        results = {"a": self._make_results(0.90, "a")}
        gr = check_gate(results, gate)
        summary = gr.summary()
        assert "PASSED" in summary
        assert "TEST" in summary


# ---------------------------------------------------------------------------
# check_gate_min_pass
# ---------------------------------------------------------------------------

class TestCheckGateMinPass:
    def _make_results(self, scores: dict[str, float]) -> dict[str, EvalResults]:
        out = {}
        for task, score in scores.items():
            out[task] = EvalResults(
                test_id=task, n_trials=10, n_success=int(score * 10),
                score=score, ci_lower=score - 0.1, ci_upper=score + 0.1, passed=score >= 0.5,
            )
        return out

    def test_checkpoint_a_6_of_10(self):
        # Provide 6 tasks above threshold, 4 below
        scores = {
            "stimulus_response": 0.85,
            "temporal_contingency": 0.55,
            "extinction": 0.75,
            "spontaneous_recovery": 0.55,
            "generalization": 0.75,
            "discrimination": 0.65,
            "reward_contingency": 0.30,   # below 1.00
            "partial_reinforcement": 0.30, # below 1.00
            "shaping": 0.30,              # below 0.60
            "chaining": 0.30,             # below 0.70
        }
        results = self._make_results(scores)
        gr = check_gate_min_pass(results, CHECKPOINT_A, min_passing=6)
        assert gr.passed is True

    def test_checkpoint_a_fails_below_6(self):
        scores = {k: 0.30 for k in CHECKPOINT_A.criteria}
        results = self._make_results(scores)
        gr = check_gate_min_pass(results, CHECKPOINT_A, min_passing=6)
        assert gr.passed is False

    def test_min_passing_exact_boundary(self):
        gate = GateConfig(name="G", criteria={"a": 0.5, "b": 0.5, "c": 0.5})
        results = {
            "a": EvalResults("a", 10, 6, 0.6, 0.4, 0.8, True),
            "b": EvalResults("b", 10, 4, 0.4, 0.2, 0.6, False),
            "c": EvalResults("c", 10, 4, 0.4, 0.2, 0.6, False),
        }
        gr = check_gate_min_pass(results, gate, min_passing=1)
        assert gr.passed is True
        gr2 = check_gate_min_pass(results, gate, min_passing=2)
        assert gr2.passed is False


# ---------------------------------------------------------------------------
# Predefined gates exist and have expected structure
# ---------------------------------------------------------------------------

class TestPredefinedGates:
    def test_checkpoint_a_has_10_criteria(self):
        assert len(CHECKPOINT_A.criteria) == 10

    def test_checkpoint_b_has_tc11(self):
        assert "reflex_validation" in CHECKPOINT_B.criteria

    def test_dino_probe_gate(self):
        assert "dino_probe" in DINO_PROBE_GATE.criteria
        assert DINO_PROBE_GATE.criteria["dino_probe"] == pytest.approx(0.70)

    def test_checkpoint_d_has_14_criteria(self):
        assert len(CHECKPOINT_D.criteria) == 14
