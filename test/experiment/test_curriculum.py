# Tests for slm_lab/experiment/curriculum.py
import json
import os
import tempfile

import pytest

from slm_lab.experiment.curriculum import (
    MASTERY_THRESHOLD,
    MASTERY_WINDOW,
    PAVLOVIAN_TASKS,
    SENSORIMOTOR_TASKS,
    TASK_THRESHOLDS,
    CurriculumSequencer,
    CurriculumState,
    Stage,
    TaskRecord,
    check_mastery,
)
from slm_lab.experiment.eval import EvalResults
from slm_lab.experiment.gates import CHECKPOINT_A, CHECKPOINT_D


# ---------------------------------------------------------------------------
# check_mastery
# ---------------------------------------------------------------------------

class TestCheckMastery:
    def test_short_history_never_masters(self):
        assert not check_mastery([1.0] * (MASTERY_WINDOW - 1))

    def test_exact_window_at_threshold(self):
        scores = [MASTERY_THRESHOLD] * MASTERY_WINDOW
        assert check_mastery(scores)

    def test_just_below_threshold(self):
        scores = [MASTERY_THRESHOLD - 0.01] * MASTERY_WINDOW
        assert not check_mastery(scores)

    def test_window_uses_only_last_n(self):
        # Lots of zeros followed by enough good scores
        bad = [0.0] * 100
        good = [1.0] * MASTERY_WINDOW
        assert check_mastery(bad + good)

    def test_custom_threshold_and_window(self):
        assert check_mastery([0.9] * 5, threshold=0.9, window=5)
        assert not check_mastery([0.9] * 4, threshold=0.9, window=5)

    def test_partial_window_fails(self):
        # 19 good scores — still one short of window=20
        assert not check_mastery([1.0] * 19)


# ---------------------------------------------------------------------------
# CurriculumState serialisation
# ---------------------------------------------------------------------------

class TestCurriculumState:
    def test_round_trip_empty(self):
        state = CurriculumState()
        d = state.to_dict()
        restored = CurriculumState.from_dict(d)
        assert restored.current_stage == state.current_stage
        assert restored.current_task_idx == state.current_task_idx
        assert restored.global_episode == state.global_episode

    def test_round_trip_with_task_records(self):
        state = CurriculumState()
        state.task_records["stimulus_response"] = TaskRecord(
            name="stimulus_response",
            stage="pavlovian",
            attempts=50,
            mastered=True,
            score_history=[0.8] * 20,
            first_mastered_at=50,
        )
        d = state.to_dict()
        restored = CurriculumState.from_dict(d)
        rec = restored.task_records["stimulus_response"]
        assert rec.mastered is True
        assert rec.attempts == 50
        assert rec.first_mastered_at == 50
        assert len(rec.score_history) == 20

    def test_save_and_load(self):
        state = CurriculumState(global_episode=42)
        state.task_records["chaining"] = TaskRecord(
            name="chaining", stage="pavlovian", attempts=10
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state.save(path)
            restored = CurriculumState.load(path)
            assert restored.global_episode == 42
            assert "chaining" in restored.task_records
        finally:
            os.unlink(path)

    def test_file_is_valid_json(self):
        state = CurriculumState(current_stage=Stage.SENSORIMOTOR.value)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            state.save(path)
            with open(path) as fp:
                data = json.load(fp)
            assert data["current_stage"] == "sensorimotor"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Task lists and thresholds
# ---------------------------------------------------------------------------

class TestTaskLists:
    def test_pavlovian_10_tasks(self):
        assert len(PAVLOVIAN_TASKS) == 10

    def test_sensorimotor_14_tasks(self):
        assert len(SENSORIMOTOR_TASKS) == 14

    def test_no_duplicates(self):
        all_tasks = PAVLOVIAN_TASKS + SENSORIMOTOR_TASKS
        assert len(all_tasks) == len(set(all_tasks))

    def test_thresholds_cover_all_tasks(self):
        for task in PAVLOVIAN_TASKS + SENSORIMOTOR_TASKS:
            assert task in TASK_THRESHOLDS, f"Missing threshold for {task}"

    def test_pavlovian_thresholds_match_checkpoint_a(self):
        """TASK_THRESHOLDS must agree with CHECKPOINT_A for all shared keys."""
        for task, thresh in CHECKPOINT_A.criteria.items():
            assert TASK_THRESHOLDS.get(task) == pytest.approx(thresh), task

    def test_sensorimotor_thresholds_match_checkpoint_d(self):
        """TASK_THRESHOLDS must agree with CHECKPOINT_D for all shared keys."""
        for task, thresh in CHECKPOINT_D.criteria.items():
            if task in TASK_THRESHOLDS:
                assert TASK_THRESHOLDS[task] == pytest.approx(thresh), task


# ---------------------------------------------------------------------------
# CurriculumSequencer — progression
# ---------------------------------------------------------------------------

class TestCurriculumProgression:
    def _make_seq(self, max_attempts: int = 1000) -> CurriculumSequencer:
        return CurriculumSequencer(
            max_attempts_per_task=max_attempts,
            mastery_threshold=MASTERY_THRESHOLD,
            mastery_window=MASTERY_WINDOW,
        )

    def test_initial_task_is_first_pavlovian(self):
        seq = self._make_seq()
        assert seq.current_task == PAVLOVIAN_TASKS[0]
        assert seq.current_stage == Stage.PAVLOVIAN

    def test_mastery_advances_task(self):
        seq = self._make_seq()
        task = seq.current_task
        for _ in range(MASTERY_WINDOW):
            seq.record_episode(task, 1.0)
        advanced = seq.advance_if_ready()
        assert advanced
        assert seq.current_task == PAVLOVIAN_TASKS[1]

    def test_no_advance_before_mastery_window(self):
        seq = self._make_seq()
        task = seq.current_task
        for _ in range(MASTERY_WINDOW - 1):
            seq.record_episode(task, 1.0)
        advanced = seq.advance_if_ready()
        assert not advanced
        assert seq.current_task == PAVLOVIAN_TASKS[0]

    def test_advance_through_all_pavlovian_tasks(self):
        seq = self._make_seq()
        for task in PAVLOVIAN_TASKS:
            assert seq.current_task == task
            for _ in range(MASTERY_WINDOW):
                seq.record_episode(task, 1.0)
            seq.advance_if_ready()
        # After all 10 Pavlovian tasks mastered, stage advances to Sensorimotor
        assert seq.current_stage == Stage.SENSORIMOTOR
        assert seq.current_task == SENSORIMOTOR_TASKS[0]

    def test_advance_through_all_tasks_to_complete(self):
        seq = self._make_seq()
        all_tasks = PAVLOVIAN_TASKS + SENSORIMOTOR_TASKS
        for task in all_tasks:
            assert seq.current_task == task
            for _ in range(MASTERY_WINDOW):
                seq.record_episode(task, 1.0)
            seq.advance_if_ready()
        assert seq.current_stage == Stage.COMPLETE
        assert seq.current_task is None

    def test_global_episode_increments(self):
        seq = self._make_seq()
        task = seq.current_task
        seq.record_episode(task, 0.5)
        seq.record_episode(task, 0.5)
        assert seq.state.global_episode == 2

    def test_no_advance_after_complete(self):
        seq = self._make_seq()
        seq.state.current_stage = Stage.COMPLETE.value
        advanced = seq.advance_if_ready()
        assert not advanced


# ---------------------------------------------------------------------------
# CurriculumSequencer — mastery detection on task record
# ---------------------------------------------------------------------------

class TestMasteryDetection:
    def test_mastery_flag_set_on_record(self):
        seq = CurriculumSequencer()
        task = PAVLOVIAN_TASKS[0]
        for _ in range(MASTERY_WINDOW):
            seq.record_episode(task, 1.0)
        rec = seq.state.task_records[task]
        assert rec.mastered is True
        assert rec.first_mastered_at is not None

    def test_mastery_flag_not_set_below_threshold(self):
        seq = CurriculumSequencer()
        task = PAVLOVIAN_TASKS[0]
        for _ in range(MASTERY_WINDOW * 2):
            seq.record_episode(task, 0.79)
        rec = seq.state.task_records[task]
        assert rec.mastered is False

    def test_mastery_persists_after_bad_episodes(self):
        seq = CurriculumSequencer()
        task = PAVLOVIAN_TASKS[0]
        for _ in range(MASTERY_WINDOW):
            seq.record_episode(task, 1.0)
        assert seq.state.task_records[task].mastered is True
        # Recording bad scores does not un-master
        for _ in range(5):
            seq.record_episode(task, 0.0)
        assert seq.state.task_records[task].mastered is True


# ---------------------------------------------------------------------------
# CurriculumSequencer — stuck / fallback
# ---------------------------------------------------------------------------

class TestStuckFallback:
    def test_stuck_after_max_attempts(self):
        seq = CurriculumSequencer(max_attempts_per_task=10)
        task = seq.current_task
        for _ in range(10):
            seq.record_episode(task, 0.0)  # always failing
        advanced = seq.advance_if_ready()
        assert advanced
        rec = seq.state.task_records[task]
        assert rec.flagged_stuck is True

    def test_stuck_task_advances_to_next(self):
        seq = CurriculumSequencer(max_attempts_per_task=5)
        first_task = seq.current_task
        for _ in range(5):
            seq.record_episode(first_task, 0.0)
        seq.advance_if_ready()
        assert seq.current_task == PAVLOVIAN_TASKS[1]

    def test_stuck_flag_does_not_affect_next_task(self):
        seq = CurriculumSequencer(max_attempts_per_task=5)
        first_task = seq.current_task
        for _ in range(5):
            seq.record_episode(first_task, 0.0)
        seq.advance_if_ready()
        second_task = seq.current_task
        assert not seq.state.task_records[second_task].flagged_stuck

    def test_mastery_before_max_attempts_no_stuck_flag(self):
        seq = CurriculumSequencer(max_attempts_per_task=1000)
        task = seq.current_task
        for _ in range(MASTERY_WINDOW):
            seq.record_episode(task, 1.0)
        seq.advance_if_ready()
        assert not seq.state.task_records[task].flagged_stuck


# ---------------------------------------------------------------------------
# CurriculumSequencer — stage boundary / EWC hook
# ---------------------------------------------------------------------------

class TestStageBoundary:
    def test_ewc_hook_called_at_pavlovian_exit(self):
        hook_calls: list[tuple] = []

        def hook(agent, stage_name):
            hook_calls.append((agent, stage_name))

        seq = CurriculumSequencer(ewc_snapshot_hook=hook)
        for task in PAVLOVIAN_TASKS:
            for _ in range(MASTERY_WINDOW):
                seq.record_episode(task, 1.0)
            seq.advance_if_ready()

        assert len(hook_calls) == 1
        assert hook_calls[0][1] == Stage.PAVLOVIAN.value

    def test_ewc_hook_called_at_sensorimotor_exit(self):
        hook_calls: list[tuple] = []

        def hook(agent, stage_name):
            hook_calls.append((agent, stage_name))

        seq = CurriculumSequencer(ewc_snapshot_hook=hook)
        all_tasks = PAVLOVIAN_TASKS + SENSORIMOTOR_TASKS
        for task in all_tasks:
            for _ in range(MASTERY_WINDOW):
                seq.record_episode(task, 1.0)
            seq.advance_if_ready()

        assert len(hook_calls) == 2
        assert hook_calls[1][1] == Stage.SENSORIMOTOR.value

    def test_ewc_hook_exception_does_not_crash_curriculum(self):
        def bad_hook(agent, stage_name):
            raise RuntimeError("hook error")

        seq = CurriculumSequencer(ewc_snapshot_hook=bad_hook)
        for task in PAVLOVIAN_TASKS:
            for _ in range(MASTERY_WINDOW):
                seq.record_episode(task, 1.0)
            seq.advance_if_ready()  # should not raise

        assert seq.current_stage == Stage.SENSORIMOTOR

    def test_stage_transitions_reset_task_idx(self):
        seq = CurriculumSequencer()
        for task in PAVLOVIAN_TASKS:
            for _ in range(MASTERY_WINDOW):
                seq.record_episode(task, 1.0)
            seq.advance_if_ready()
        assert seq.state.current_task_idx == 0
        assert seq.current_task == SENSORIMOTOR_TASKS[0]

    def test_completed_at_set_on_completion(self):
        seq = CurriculumSequencer()
        for task in PAVLOVIAN_TASKS + SENSORIMOTOR_TASKS:
            for _ in range(MASTERY_WINDOW):
                seq.record_episode(task, 1.0)
            seq.advance_if_ready()
        assert seq.state.completed_at is not None


# ---------------------------------------------------------------------------
# CurriculumSequencer — gate integration
# ---------------------------------------------------------------------------

class TestGateIntegration:
    def _make_eval_result(self, task: str, score: float, threshold: float | None = None) -> EvalResults:
        if threshold is None:
            threshold = TASK_THRESHOLDS.get(task, 0.5)
        return EvalResults(
            test_id=task,
            n_trials=20,
            n_success=int(score * 20),
            score=score,
            ci_lower=max(0.0, score - 0.1),
            ci_upper=min(1.0, score + 0.1),
            passed=score >= threshold,
        )

    def test_gate_passes_when_6_of_10_pavlovian_pass(self):
        seq = CurriculumSequencer()
        seq.state.current_stage = Stage.PAVLOVIAN.value
        # Store eval results for tasks above threshold (6 of 10)
        passing_tasks = PAVLOVIAN_TASKS[:6]
        failing_tasks = PAVLOVIAN_TASKS[6:]
        for task in passing_tasks:
            result = self._make_eval_result(task, TASK_THRESHOLDS[task] + 0.05)
            seq.record_eval_result(task, result)
        for task in failing_tasks:
            result = self._make_eval_result(task, 0.0)
            seq.record_eval_result(task, result)
        assert seq.run_gate_check() is True

    def test_gate_fails_when_fewer_than_6_pavlovian_pass(self):
        seq = CurriculumSequencer()
        seq.state.current_stage = Stage.PAVLOVIAN.value
        for task in PAVLOVIAN_TASKS:
            result = self._make_eval_result(task, 0.0)
            seq.record_eval_result(task, result)
        assert seq.run_gate_check() is False

    def test_sensorimotor_gate_check(self):
        seq = CurriculumSequencer()
        seq.state.current_stage = Stage.SENSORIMOTOR.value
        # Store results under CHECKPOINT_D criteria keys (gate expects these names)
        for task, threshold in CHECKPOINT_D.criteria.items():
            result = self._make_eval_result(task, threshold + 0.05, threshold=threshold)
            seq.record_eval_result(task, result)
        assert seq.run_gate_check() is True

    def test_sensorimotor_gate_fails_with_missing_tasks(self):
        seq = CurriculumSequencer()
        seq.state.current_stage = Stage.SENSORIMOTOR.value
        # Only provide half the CHECKPOINT_D tasks
        keys = list(CHECKPOINT_D.criteria.keys())
        for task in keys[:7]:
            result = self._make_eval_result(task, 1.0)
            seq.record_eval_result(task, result)
        assert seq.run_gate_check() is False

    def test_record_eval_result_persisted_in_state(self):
        seq = CurriculumSequencer()
        result = self._make_eval_result("stimulus_response", 0.90)
        seq.record_eval_result("stimulus_response", result)
        snap = seq.state.stage_eval_results["stimulus_response"]
        assert snap["score"] == pytest.approx(0.90)
        assert snap["passed"] is True


# ---------------------------------------------------------------------------
# CurriculumSequencer — checkpoint / resume
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    def test_save_and_resume_preserves_progress(self):
        seq = CurriculumSequencer(max_attempts_per_task=1000)
        # Advance through 3 tasks
        for task in PAVLOVIAN_TASKS[:3]:
            for _ in range(MASTERY_WINDOW):
                seq.record_episode(task, 1.0)
            seq.advance_if_ready()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            seq.save_state(path)
            seq2 = CurriculumSequencer()
            seq2.load_state(path)
            assert seq2.state.current_task_idx == seq.state.current_task_idx
            assert seq2.state.global_episode == seq.state.global_episode
            assert seq2.current_task == seq.current_task
            # Mastery flags preserved
            for task in PAVLOVIAN_TASKS[:3]:
                assert seq2.state.task_records[task].mastered is True
        finally:
            os.unlink(path)

    def test_resume_continues_training_correctly(self):
        seq = CurriculumSequencer()
        first_task = seq.current_task
        # Do partial training (not yet mastered)
        for _ in range(5):
            seq.record_episode(first_task, 0.5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            seq.save_state(path)
            seq2 = CurriculumSequencer()
            seq2.load_state(path)
            # Resume: add more scores to reach mastery
            for _ in range(MASTERY_WINDOW):
                seq2.record_episode(first_task, 1.0)
            assert seq2.state.task_records[first_task].mastered is True
        finally:
            os.unlink(path)

    def test_stuck_flag_preserved_across_checkpoint(self):
        seq = CurriculumSequencer(max_attempts_per_task=5)
        task = seq.current_task
        for _ in range(5):
            seq.record_episode(task, 0.0)
        seq.advance_if_ready()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            seq.save_state(path)
            seq2 = CurriculumSequencer()
            seq2.load_state(path)
            assert seq2.state.task_records[task].flagged_stuck is True
        finally:
            os.unlink(path)

    def test_stage_preserved_across_checkpoint(self):
        seq = CurriculumSequencer()
        for task in PAVLOVIAN_TASKS:
            for _ in range(MASTERY_WINDOW):
                seq.record_episode(task, 1.0)
            seq.advance_if_ready()
        assert seq.current_stage == Stage.SENSORIMOTOR

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            seq.save_state(path)
            seq2 = CurriculumSequencer()
            seq2.load_state(path)
            assert seq2.current_stage == Stage.SENSORIMOTOR
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Summary smoke test
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_contains_stage_and_task(self):
        seq = CurriculumSequencer()
        text = seq.summary()
        assert "pavlovian" in text.lower()
        assert PAVLOVIAN_TASKS[0] in text

    def test_summary_shows_mastered_flag(self):
        seq = CurriculumSequencer()
        task = seq.current_task
        for _ in range(MASTERY_WINDOW):
            seq.record_episode(task, 1.0)
        seq.advance_if_ready()
        text = seq.summary()
        assert "MASTERED" in text

    def test_summary_shows_stuck_flag(self):
        seq = CurriculumSequencer(max_attempts_per_task=3)
        task = seq.current_task
        for _ in range(3):
            seq.record_episode(task, 0.0)
        seq.advance_if_ready()
        text = seq.summary()
        assert "STUCK" in text
