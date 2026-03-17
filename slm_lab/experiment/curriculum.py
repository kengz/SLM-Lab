# Curriculum sequencer for Turing Curriculum TC-01 through TC-24.
# Progresses tasks in order, declares mastery via rolling success window,
# handles stuck detection, and fires stage-transition hooks.
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable

from slm_lab.experiment.eval import EvalResults, run_eval
from slm_lab.experiment.gates import (
    GateConfig,
    check_gate,
    check_gate_min_pass,
    CHECKPOINT_A,
    CHECKPOINT_D,
)
from slm_lab.lib import logger

_log = logger.get_logger(__name__)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

class Stage(str, Enum):
    PAVLOVIAN = "pavlovian"
    SENSORIMOTOR = "sensorimotor"
    COMPLETE = "complete"


# Ordered task lists per stage (names match gates.py criteria keys)
PAVLOVIAN_TASKS: list[str] = [
    "stimulus_response",      # TC-01
    "temporal_contingency",   # TC-02
    "extinction",             # TC-03
    "spontaneous_recovery",   # TC-04
    "generalization",         # TC-05
    "discrimination",         # TC-06
    "reward_contingency",     # TC-07
    "partial_reinforcement",  # TC-08
    "shaping",                # TC-09
    "chaining",               # TC-10
]

SENSORIMOTOR_TASKS: list[str] = [
    "reflex_validation",          # TC-11 — born-ready reflexes
    "contingency_detection",      # TC-12 — action-effect discovery
    "reach_grasp",                # TC-13 — motor coordination / reaching
    "object_permanence_basic",    # TC-14 — object interaction
    "means_ends",                 # TC-15 — means-end precursor
    "ab_error",                   # TC-16 — object permanence A-not-B
    "spatial_reasoning",          # TC-17 — intentional means-end
    "tool_use_proximal",          # TC-18 — tool use cloth
    "imitation",                  # TC-19 — secondary circular imitation
    "object_categorization",      # TC-20 — novel tool use / categorization
    "tool_use_distal",            # TC-21 — distal tool use
    "insight",                    # TC-22 — insightful problem solving
    "working_memory",             # TC-23 — deferred imitation / working memory
    "object_permanence_advanced", # TC-24 — invisible displacement
]

# Per-task pass thresholds (sourced from gates.py + test specs).
# Sensorimotor names are aligned with CHECKPOINT_D criteria keys.
TASK_THRESHOLDS: dict[str, float] = {
    # Pavlovian (CHECKPOINT_A)
    "stimulus_response": 0.80,
    "temporal_contingency": 0.50,
    "extinction": 0.70,
    "spontaneous_recovery": 0.50,
    "generalization": 0.70,
    "discrimination": 0.60,
    "reward_contingency": 1.00,
    "partial_reinforcement": 1.00,
    "shaping": 0.60,
    "chaining": 0.70,
    # Sensorimotor (CHECKPOINT_D)
    "reflex_validation": 0.90,
    "contingency_detection": 0.60,
    "reach_grasp": 0.50,
    "object_permanence_basic": 0.50,
    "means_ends": 0.60,
    "ab_error": 0.60,
    "spatial_reasoning": 0.60,
    "tool_use_proximal": 0.60,
    "imitation": 0.70,
    "object_categorization": 0.50,
    "tool_use_distal": 0.56,
    "insight": 0.45,
    "working_memory": 0.50,
    "object_permanence_advanced": 0.55,
}

# Mastery parameters
MASTERY_THRESHOLD: float = 0.80
MASTERY_WINDOW: int = 20


# ---------------------------------------------------------------------------
# Curriculum state (serializable)
# ---------------------------------------------------------------------------

@dataclass
class TaskRecord:
    """Rolling history and status for one task."""
    name: str
    stage: str
    attempts: int = 0                        # total episodes attempted on this task
    mastered: bool = False
    flagged_stuck: bool = False              # advanced due to max_attempts
    score_history: list[float] = field(default_factory=list)  # per-episode scores
    first_mastered_at: int | None = None     # episode index when mastery first declared
    last_eval_result: dict | None = None     # serialised EvalResults snapshot


@dataclass
class CurriculumState:
    """Full serialisable curriculum state for checkpoint/resume."""
    current_stage: str = Stage.PAVLOVIAN.value
    current_task_idx: int = 0                # index within the active stage task list
    global_episode: int = 0                  # total episodes across all tasks
    task_records: dict[str, TaskRecord] = field(default_factory=dict)
    stage_eval_results: dict[str, dict] = field(default_factory=dict)  # task -> EvalResults dict
    pavlovian_gate_passed: bool = False
    sensorimotor_gate_passed: bool = False
    completed_at: float | None = None        # wall time when COMPLETE reached

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CurriculumState":
        state = cls(
            current_stage=d["current_stage"],
            current_task_idx=d["current_task_idx"],
            global_episode=d["global_episode"],
            stage_eval_results=d.get("stage_eval_results", {}),
            pavlovian_gate_passed=d.get("pavlovian_gate_passed", False),
            sensorimotor_gate_passed=d.get("sensorimotor_gate_passed", False),
            completed_at=d.get("completed_at"),
        )
        for name, rec in d.get("task_records", {}).items():
            state.task_records[name] = TaskRecord(**rec)
        return state

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        _log.info(f"CurriculumState saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CurriculumState":
        with open(path) as f:
            d = json.load(f)
        _log.info(f"CurriculumState loaded from {path}")
        return cls.from_dict(d)


# ---------------------------------------------------------------------------
# Mastery detection
# ---------------------------------------------------------------------------

def check_mastery(
    score_history: list[float],
    threshold: float = MASTERY_THRESHOLD,
    window: int = MASTERY_WINDOW,
) -> bool:
    """Return True if rolling mean of the last `window` scores >= threshold."""
    if len(score_history) < window:
        return False
    recent = score_history[-window:]
    return (sum(recent) / len(recent)) >= threshold


# ---------------------------------------------------------------------------
# Main sequencer
# ---------------------------------------------------------------------------

class CurriculumSequencer:
    """Sequences tasks TC-01 through TC-24 with mastery detection.

    Args:
        max_attempts_per_task: Episodes before flagging a task stuck and
            advancing regardless of mastery.
        mastery_threshold: Rolling mean score required for mastery.
        mastery_window: Number of recent episodes for rolling mean.
        ewc_snapshot_hook: Called with (agent, stage_name) at each stage
            transition. Intended for EWC Fisher snapshot capture.
        eval_every: Periodically run formal eval every N episodes per task
            (0 = never run formal eval automatically).
        min_passing_pavlovian: Minimum Pavlovian tasks that must pass the
            gate check before advancing to sensorimotor stage.
    """

    def __init__(
        self,
        max_attempts_per_task: int = 5000,
        mastery_threshold: float = MASTERY_THRESHOLD,
        mastery_window: int = MASTERY_WINDOW,
        ewc_snapshot_hook: Callable | None = None,
        eval_every: int = 0,
        min_passing_pavlovian: int = 6,
    ) -> None:
        self.max_attempts_per_task = max_attempts_per_task
        self.mastery_threshold = mastery_threshold
        self.mastery_window = mastery_window
        self.ewc_snapshot_hook = ewc_snapshot_hook
        self.eval_every = eval_every
        self.min_passing_pavlovian = min_passing_pavlovian
        self.state = CurriculumState()
        self._init_task_records()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_task_records(self) -> None:
        for name in PAVLOVIAN_TASKS:
            if name not in self.state.task_records:
                self.state.task_records[name] = TaskRecord(
                    name=name, stage=Stage.PAVLOVIAN.value
                )
        for name in SENSORIMOTOR_TASKS:
            if name not in self.state.task_records:
                self.state.task_records[name] = TaskRecord(
                    name=name, stage=Stage.SENSORIMOTOR.value
                )

    def _tasks_for_stage(self, stage: Stage) -> list[str]:
        if stage == Stage.PAVLOVIAN:
            return PAVLOVIAN_TASKS
        if stage == Stage.SENSORIMOTOR:
            return SENSORIMOTOR_TASKS
        return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> Stage:
        return Stage(self.state.current_stage)

    @property
    def current_task(self) -> str | None:
        """Name of the active task, or None when complete."""
        if self.current_stage == Stage.COMPLETE:
            return None
        tasks = self._tasks_for_stage(self.current_stage)
        idx = self.state.current_task_idx
        if idx >= len(tasks):
            return None
        return tasks[idx]

    def record_episode(self, task_name: str, score: float) -> None:
        """Record one training episode score for the given task.

        Call this after every training episode. The sequencer updates the
        task record and checks mastery; call `advance_if_ready()` after
        to handle task/stage transitions.
        """
        rec = self.state.task_records[task_name]
        rec.score_history.append(score)
        rec.attempts += 1
        self.state.global_episode += 1

        if not rec.mastered and check_mastery(
            rec.score_history, self.mastery_threshold, self.mastery_window
        ):
            rec.mastered = True
            rec.first_mastered_at = self.state.global_episode
            _log.info(
                f"[curriculum] {task_name} MASTERED at episode "
                f"{self.state.global_episode} (rolling mean >= {self.mastery_threshold})"
            )

    def advance_if_ready(self, agent=None) -> bool:
        """Check current task and advance to next if mastered or stuck.

        Returns True if a task/stage transition happened.
        """
        if self.current_stage == Stage.COMPLETE:
            return False

        task = self.current_task
        if task is None:
            return False

        rec = self.state.task_records[task]
        should_advance = False
        reason = ""

        if rec.mastered:
            should_advance = True
            reason = "mastered"
        elif rec.attempts >= self.max_attempts_per_task:
            rec.flagged_stuck = True
            should_advance = True
            reason = f"stuck (attempts={rec.attempts} >= max={self.max_attempts_per_task})"
            _log.warning(
                f"[curriculum] {task} stuck after {rec.attempts} attempts — "
                "advancing with flag"
            )

        if should_advance:
            _log.info(f"[curriculum] advancing past {task} ({reason})")
            self._advance_task(agent)
            return True

        return False

    def record_eval_result(self, task_name: str, result: EvalResults) -> None:
        """Store a formal EvalResults snapshot for a task.

        task_name may be any registered training task or a gate criteria key.
        """
        snap = {
            "test_id": result.test_id,
            "score": result.score,
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "passed": result.passed,
            "n_trials": result.n_trials,
        }
        self.state.stage_eval_results[task_name] = snap
        # Update TaskRecord if this task is registered
        if task_name in self.state.task_records:
            self.state.task_records[task_name].last_eval_result = snap

    def run_gate_check(self) -> bool:
        """Run formal gate check for the current stage using stored eval results.

        Returns True if the gate passes.
        """
        stage = self.current_stage
        results = self._reconstruct_eval_results()

        if stage == Stage.PAVLOVIAN:
            gr = check_gate_min_pass(results, CHECKPOINT_A, self.min_passing_pavlovian)
            self.state.pavlovian_gate_passed = gr.passed
            _log.info(f"[curriculum] Pavlovian gate check: {gr.summary()}")
            return gr.passed

        if stage == Stage.SENSORIMOTOR:
            gr = check_gate(results, CHECKPOINT_D)
            self.state.sensorimotor_gate_passed = gr.passed
            _log.info(f"[curriculum] Sensorimotor gate check: {gr.summary()}")
            return gr.passed

        return False

    def load_state(self, path: str) -> None:
        """Restore curriculum from a checkpoint file."""
        self.state = CurriculumState.load(path)
        self._init_task_records()

    def save_state(self, path: str) -> None:
        """Persist current curriculum state to a checkpoint file."""
        self.state.save(path)

    def summary(self) -> str:
        """Return a human-readable progress summary."""
        lines = [
            f"Stage: {self.state.current_stage}  "
            f"Task: {self.current_task}  "
            f"Global episode: {self.state.global_episode}",
        ]
        for stage_tasks in (PAVLOVIAN_TASKS, SENSORIMOTOR_TASKS):
            for name in stage_tasks:
                rec = self.state.task_records.get(name)
                if rec is None:
                    continue
                flags = []
                if rec.mastered:
                    flags.append("MASTERED")
                if rec.flagged_stuck:
                    flags.append("STUCK")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                lines.append(f"  {name}: attempts={rec.attempts}{flag_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_task(self, agent=None) -> None:
        """Move to the next task, handling stage boundaries."""
        stage = self.current_stage
        tasks = self._tasks_for_stage(stage)
        next_idx = self.state.current_task_idx + 1

        if next_idx < len(tasks):
            self.state.current_task_idx = next_idx
            next_task = tasks[next_idx]
            _log.info(f"[curriculum] next task: {next_task} (stage={stage.value})")
        else:
            # All tasks in this stage exhausted — attempt stage transition
            self._transition_stage(agent)

    def _transition_stage(self, agent=None) -> None:
        """Transition from current stage to the next."""
        current = self.current_stage

        if current == Stage.PAVLOVIAN:
            _log.info("[curriculum] Pavlovian stage complete — transitioning to Sensorimotor")
            if self.ewc_snapshot_hook is not None:
                try:
                    self.ewc_snapshot_hook(agent, Stage.PAVLOVIAN.value)
                except Exception as exc:
                    _log.error(f"[curriculum] EWC snapshot hook failed: {exc}")
            self.state.current_stage = Stage.SENSORIMOTOR.value
            self.state.current_task_idx = 0
            _log.info(
                f"[curriculum] first sensorimotor task: {SENSORIMOTOR_TASKS[0]}"
            )

        elif current == Stage.SENSORIMOTOR:
            _log.info("[curriculum] Sensorimotor stage complete — curriculum DONE")
            if self.ewc_snapshot_hook is not None:
                try:
                    self.ewc_snapshot_hook(agent, Stage.SENSORIMOTOR.value)
                except Exception as exc:
                    _log.error(f"[curriculum] EWC snapshot hook failed: {exc}")
            self.state.current_stage = Stage.COMPLETE.value
            self.state.current_task_idx = 0
            self.state.completed_at = time.time()

    def _reconstruct_eval_results(self) -> dict[str, EvalResults]:
        """Rebuild EvalResults dict from stored snapshots for gate checks."""
        out: dict[str, EvalResults] = {}
        for task_name, snap in self.state.stage_eval_results.items():
            out[task_name] = EvalResults(
                test_id=snap["test_id"],
                n_trials=snap["n_trials"],
                n_success=int(snap["score"] * snap["n_trials"]),
                score=snap["score"],
                ci_lower=snap["ci_lower"],
                ci_upper=snap["ci_upper"],
                passed=snap["passed"],
            )
        return out
