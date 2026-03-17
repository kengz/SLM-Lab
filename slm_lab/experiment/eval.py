# Evaluation runner for Turing Curriculum (TC) tests.
# Scoring functions are pure; this module handles trial management and statistics.
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.stats import beta as beta_dist

from slm_lab.lib import logger as _logger_module

logger = _logger_module.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    test_id: str
    n_trials: int
    n_success: int
    score: float                    # mean score across trials [0, 1]
    ci_lower: float                 # 95% CI lower bound
    ci_upper: float                 # 95% CI upper bound
    passed: bool                    # score >= threshold AND ci_lower >= ci_threshold
    trial_scores: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)   # aggregated task-specific metrics


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def clopper_pearson_ci(
    successes: int,
    trials: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Exact binomial CI (Clopper-Pearson). Never undercovers."""
    if trials == 0:
        return (0.0, 1.0)
    # Lower bound: undefined when successes == 0 → clamp to 0.0
    lo = 0.0 if successes == 0 else float(beta_dist.ppf(alpha / 2, successes, trials - successes + 1))
    # Upper bound: undefined when successes == trials → clamp to 1.0
    hi = 1.0 if successes == trials else float(beta_dist.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return (lo, hi)


def bootstrap_ci(
    scores: Sequence[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for continuous scores."""
    rng = np.random.default_rng(seed)
    arr = np.array(scores, dtype=float)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def compute_ci(
    scores: Sequence[float],
    score_type: str = "binary",
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Select CI method by score_type: 'binary' (Clopper-Pearson) or 'continuous' (bootstrap)."""
    if score_type == "binary":
        successes = sum(1 for s in scores if s >= 0.5)
        return clopper_pearson_ci(successes, len(scores), alpha)
    return bootstrap_ci(scores, alpha=alpha)


def check_threshold(
    results: "EvalResults",
    threshold: float,
    ci_threshold: float | None = None,
) -> bool:
    """Return True if score >= threshold and (if provided) ci_lower >= ci_threshold."""
    if results.score < threshold:
        return False
    if ci_threshold is not None and results.ci_lower < ci_threshold:
        return False
    return True


def iqm(scores: Sequence[float]) -> float:
    """Interquartile mean: mean of the middle 50% (rliable, Agarwal et al. 2021)."""
    arr = np.sort(np.array(scores, dtype=float))
    n = len(arr)
    lo = n // 4
    hi = n - n // 4
    return float(arr[lo:hi].mean()) if hi > lo else float(arr.mean())


# ---------------------------------------------------------------------------
# Core eval runner
# ---------------------------------------------------------------------------

def run_eval(
    env,
    agent,
    n_trials: int = 20,
    score_type: str = "binary",
    test_id: str = "unknown",
    threshold: float = 0.5,
    ci_threshold: float | None = None,
) -> EvalResults:
    """Run n_trials evaluation episodes and return EvalResults.

    Args:
        env: Gymnasium env (single, not vectorised). Must have reset()/step().
        agent: Agent with act(obs, deterministic=True) method.
        n_trials: Number of probe episodes to run.
        score_type: "binary" or "continuous" — selects CI method.
        test_id: TC test identifier for logging.
        threshold: Pass threshold for check_threshold.
        ci_threshold: Optional CI lower-bound threshold.

    Returns:
        EvalResults with score, CI, passed flag.
    """
    trial_scores: list[float] = []
    all_metrics: list[dict] = []

    for i in range(n_trials):
        obs, info = env.reset(seed=i * 1000)
        done = False
        episode_metrics: dict = {}

        while not done:
            action = agent.act(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_metrics = info  # keep last info

        # Score is either from info["score"] or derived from info["is_success"]
        if "score" in episode_metrics:
            score = float(episode_metrics["score"])
        elif "is_success" in episode_metrics:
            score = 1.0 if episode_metrics["is_success"] else 0.0
        else:
            logger.warning(f"[{test_id}] trial {i}: no score or is_success in info; defaulting to 0.0")
            score = 0.0

        trial_scores.append(score)
        all_metrics.append(episode_metrics)

    ci_lo, ci_hi = compute_ci(trial_scores, score_type=score_type)
    mean_score = float(np.mean(trial_scores))
    n_success = sum(1 for s in trial_scores if s >= 0.5)

    aggregated = _aggregate_metrics(all_metrics)

    passed = check_threshold(
        EvalResults(test_id, n_trials, n_success, mean_score, ci_lo, ci_hi, False),
        threshold,
        ci_threshold,
    )
    results = EvalResults(
        test_id=test_id,
        n_trials=n_trials,
        n_success=n_success,
        score=mean_score,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        passed=passed,
        trial_scores=trial_scores,
        metrics=aggregated,
    )

    logger.info(
        f"[{test_id}] {n_trials} trials | score={mean_score:.3f} "
        f"CI=[{ci_lo:.3f}, {ci_hi:.3f}] | passed={results.passed}"
    )
    return results


def _aggregate_metrics(all_metrics: list[dict]) -> dict[str, float]:
    """Mean of numeric metric values across trials (union of all trial keys)."""
    if not all_metrics:
        return {}
    keys = {k for m in all_metrics for k, v in m.items() if isinstance(v, (int, float))}
    return {
        k: float(np.mean([m[k] for m in all_metrics if k in m]))
        for k in keys
    }


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def format_results(results: EvalResults) -> str:
    """Return a multi-line summary table for one EvalResults."""
    lines = [
        f"{'─' * 52}",
        f"  Test : {results.test_id}",
        f"  Score: {results.score:.3f}  (n={results.n_trials}, successes={results.n_success})",
        f"  95%CI: [{results.ci_lower:.3f}, {results.ci_upper:.3f}]",
        f"  Pass : {'YES' if results.passed else 'NO'}",
    ]
    if results.metrics:
        lines.append("  Metrics:")
        for k, v in results.metrics.items():
            if isinstance(v, float):
                lines.append(f"    {k}: {v:.4f}")
            else:
                lines.append(f"    {k}: {v}")
    lines.append(f"{'─' * 52}")
    return "\n".join(lines)
