# Phase gate system for Turing Curriculum stage advancement.
# Gates aggregate per-task EvalResults and decide whether a checkpoint is passed.
from dataclasses import dataclass, field

from slm_lab.lib import logger
from slm_lab.experiment.eval import EvalResults

logger = logger.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GateConfig:
    """Defines a phase gate as a set of (task_name -> pass_threshold) criteria."""
    name: str
    criteria: dict[str, float]          # task_name -> minimum score threshold
    description: str = ""


@dataclass
class GateResult:
    """Outcome of evaluating one GateConfig against a results dict."""
    gate_name: str
    passed: bool
    passing: dict[str, float]           # task -> score for tasks that passed
    failing: dict[str, float]           # task -> score for tasks that failed
    missing: list[str]                  # tasks with no result entry
    diagnostics: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Gate [{self.gate_name}]: {status}",
            f"  Passing ({len(self.passing)}): "
            + ", ".join(f"{k}={v:.3f}" for k, v in self.passing.items()),
            f"  Failing ({len(self.failing)}): "
            + ", ".join(f"{k}={v:.3f}" for k, v in self.failing.items()),
        ]
        if self.missing:
            lines.append(f"  Missing ({len(self.missing)}): {', '.join(self.missing)}")
        for d in self.diagnostics:
            lines.append(f"  ! {d}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def check_gate(results: dict[str, EvalResults], gate: GateConfig) -> GateResult:
    """Evaluate gate criteria against a results dict.

    Args:
        results: Mapping of task_name -> EvalResults (from run_eval or manual).
        gate: GateConfig with name and criteria dict.

    Returns:
        GateResult with pass/fail status and per-task diagnostics.
    """
    passing: dict[str, float] = {}
    failing: dict[str, float] = {}
    missing: list[str] = []
    diagnostics: list[str] = []

    for task, threshold in gate.criteria.items():
        if task not in results:
            missing.append(task)
            diagnostics.append(f"{task}: no result (not evaluated)")
            continue

        score = results[task].score
        if score >= threshold:
            passing[task] = score
        else:
            failing[task] = score
            diagnostics.append(
                f"{task}: score {score:.3f} < threshold {threshold:.3f}"
            )

    passed = len(failing) == 0 and len(missing) == 0

    gr = GateResult(
        gate_name=gate.name,
        passed=passed,
        passing=passing,
        failing=failing,
        missing=missing,
        diagnostics=diagnostics,
    )
    logger.info(gr.summary())
    return gr


# ---------------------------------------------------------------------------
# Predefined gates (Phase 3)
# ---------------------------------------------------------------------------

# Checkpoint A: Pavlovian stage — at least 6 of 10 tasks pass.
# Represented as all 10 tasks with threshold 0.0 so we can count pass counts.
# Actual ≥6/10 logic is enforced via check_gate_min_pass.
CHECKPOINT_A = GateConfig(
    name="CHECKPOINT_A",
    criteria={
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
    },
    description="Pavlovian stage exit: all 10 TC tasks at their pass thresholds",
)

# Checkpoint B: TC-11 reflex validation ≥50%.
CHECKPOINT_B = GateConfig(
    name="CHECKPOINT_B",
    criteria={"reflex_validation": 0.50},
    description="Sensorimotor stage entry: TC-11 reflex validation at ≥50%",
)

# DINO probe gate: perception probe accuracy >70%.
DINO_PROBE_GATE = GateConfig(
    name="DINO_PROBE_GATE",
    criteria={"dino_probe": 0.70},
    description="DINO perception probe: linear probe accuracy > 70%",
)

# Checkpoint D: Sensorimotor stage exit — ≥10 of 14 tasks pass, TC-24 ≥60%.
CHECKPOINT_D = GateConfig(
    name="CHECKPOINT_D",
    criteria={
        "reflex_validation": 0.90,
        "contingency_detection": 0.60,
        "reach_grasp": 0.50,
        "object_permanence_basic": 0.50,
        "imitation": 0.70,
        "ab_error": 0.60,       # expected pass in S4
        "tool_use_proximal": 0.60,
        "tool_use_distal": 0.56,
        "means_ends": 0.60,
        "spatial_reasoning": 0.60,
        "object_categorization": 0.50,
        "insight": 0.45,
        "working_memory": 0.50,
        "object_permanence_advanced": 0.55,
    },
    description="Sensorimotor stage exit: ≥10/14 tasks pass, TC-24 ≥60%",
)


def check_gate_min_pass(
    results: dict[str, EvalResults],
    gate: GateConfig,
    min_passing: int,
) -> GateResult:
    """Gate variant: passes if at least `min_passing` criteria are met.

    Used for CHECKPOINT_A (≥6/10 Pavlovian tasks).
    """
    gr = check_gate(results, gate)
    # Override pass/fail with min_passing count logic
    n_passed = len(gr.passing)
    actually_passed = n_passed >= min_passing
    diag = f"{n_passed}/{len(gate.criteria)} tasks passing (need {min_passing})"
    gr.passed = actually_passed
    gr.diagnostics.insert(0, diag)
    logger.info(f"Gate [{gate.name}] min_pass={min_passing}: {diag} → {'PASSED' if actually_passed else 'FAILED'}")
    return gr
