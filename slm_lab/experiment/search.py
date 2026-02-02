import os
import warnings
from copy import deepcopy

import numpy as np
import pydash as ps
import ray
import ray.tune as tune
import torch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from slm_lab import ROOT_DIR
from slm_lab.agent import MetricsTracker
from slm_lab.experiment.analysis import METRICS_COLS
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util

logger = logger.get_logger(__name__)

# Default meta.search spec, uses fields from tune.report()
BASE_SCHEDULER_SPEC = {
    "time_attr": "frame",
    "metric": "total_reward_ma",
    "mode": "max",
}


def in_ray_tune_context() -> bool:
    """Check if currently executing within Ray Tune trial."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ray.tune.get_context().get_trial_dir() is not None
    except Exception:
        return False


def build_param_space(spec: dict) -> dict:
    """
    Build Ray Tune parameter space from SLM-Lab spec.

    Specify a config space in spec using "{key}__{space_type}": [args] format.
    Where {space_type} is any Ray Tune search space function.

    Two argument patterns:
    - Most functions: use [arg1, arg2, ...] → tune.func(*args)
    - choice: use [item1, item2, ...] → tune.choice([items])

    Examples:
    - "gamma__uniform": [0.95, 0.999] → tune.uniform(0.95, 0.999)
    - "batch_size__choice": [16, 32, 64] → tune.choice([16, 32, 64])

    Available search space types (see https://docs.ray.io/en/latest/tune/api/search_space.html):
    - uniform(0.95, 0.999): Sample float uniformly between 0.95 and 0.999
    - quniform(3.2, 5.4, 0.2): uniform + round to multiples of 0.2
    - loguniform(1e-4, 1e-2): uniform in log space between 0.0001 and 0.01
    - qloguniform(1e-4, 1e-1, 5e-5): loguniform + round to multiples of 0.00005
    - randn(10, 2): Sample from normal distribution with mean=10, sd=2
    - qrandn(10, 2, 0.2): randn + round to multiples of 0.2
    - randint(-9, 15): Sample integer uniformly [-9, 15)
    - qrandint(-21, 12, 3): randint + round to multiples of 3 (12 inclusive)
    - lograndint(1, 10): randint in log space [1, 10)
    - qlograndint(1, 10, 2): lograndint + round to multiples of 2 (10 inclusive)
    - choice([16, 32, 64]): Sample uniformly from discrete choices
    """
    use_list_args = ("choice",)
    param_space = {}
    for k, v in util.flatten_dict(spec["search"]).items():
        key, dist = k.split("__")
        if dist == "grid_search":
            raise ValueError(
                f"grid_search is not supported with Optuna. Use 'choice' instead: {k}"
            )
        search_fn = getattr(tune, dist)
        param_space[key] = search_fn(v) if dist in use_list_args else search_fn(*v)
    return param_space


def inject_config(spec: dict, config: dict) -> dict:
    """Inject flattened config into SLM Lab spec."""
    spec = deepcopy(spec)
    spec.pop("search", None)

    # Extract trial index from Ray Tune directory (1-based → 0-based)
    trial_dir = ray.tune.get_context().get_trial_dir()
    trial_name = ray.tune.get_context().get_trial_name()
    trial_folder = trial_dir.split("/")[-1]
    ray_trial_index = int(trial_folder.replace(f"{trial_name}_", "").split("_")[0])

    # Set to ray_index - 2 so tick() increments to correct 0-based index
    spec["meta"]["trial"] = ray_trial_index - 2
    spec_util.tick(spec, "trial")

    for k, v in config.items():
        ps.set_(spec, k, v)
    return spec


def build_run_trial(base_spec: dict) -> callable:
    """Create a Ray Tune trial runner with the base spec."""

    def run_trial(config: dict) -> dict:
        from slm_lab.experiment.control import Trial

        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        spec = inject_config(base_spec, config)

        # Trial saves spec in __init__, handles both single and multi-session
        scalar_metrics = Trial(spec).run()

        # Report final metrics to Ray Tune scheduler
        if in_ray_tune_context():
            metric = scalar_metrics[BASE_SCHEDULER_SPEC["metric"]]
            metric = 0.0 if np.isnan(metric) else float(metric)
            ray.tune.report({
                BASE_SCHEDULER_SPEC["time_attr"]: spec["env"]["max_frame"],
                BASE_SCHEDULER_SPEC["metric"]: metric
            })

        return scalar_metrics

    return run_trial


def get_trial_resources(spec: dict) -> dict:
    """Build resource configuration for Ray Tune trials.

    Configurable via meta.search_resources: {"cpu": N, "gpu": N}
    Defaults: cpu=1, gpu=0 (MLP envs don't need GPU).
    """
    search_resources = spec.get("meta", {}).get("search_resources", {})
    cpu_per_trial = search_resources.get("cpu", 1)
    gpu_per_trial = search_resources.get("gpu", 0)

    # Only use GPU if available
    if gpu_per_trial > 0 and not torch.cuda.is_available():
        gpu_per_trial = 0

    return {"cpu": cpu_per_trial, "gpu": gpu_per_trial}


def extract_trial_results(results: tune.ResultGrid, spec: dict) -> dict:
    """Extract trial results from saved trial metrics files."""
    trial_results = {}
    info_dir = os.path.join(ROOT_DIR, spec['meta']['predir'], 'info')
    spec_name = spec['name']

    for i, result in enumerate(results):
        # For single-session trials, files are named with session index
        # Try both patterns: _t{i}_trial_metrics and _t{i}_s0_trial_metrics
        metrics_path = os.path.join(info_dir, f'{spec_name}_t{i}_trial_metrics_scalar.json')
        if not os.path.exists(metrics_path):
            metrics_path = os.path.join(info_dir, f'{spec_name}_t{i}_s0_trial_metrics_scalar.json')

        if os.path.exists(metrics_path):
            metrics_data = util.read(metrics_path)
        else:
            logger.warning(f"Trial {i} has no saved metrics")
            metrics_data = {col: np.nan for col in METRICS_COLS}

        trial_results[i] = {**result.config, **metrics_data}

    return trial_results


def report(mt: MetricsTracker):
    """Report metrics to Ray Tune scheduler at log_frequency."""
    if not in_ray_tune_context():
        return

    frame = mt.env.get()
    if frame >= mt.env.max_frame:
        return  # Skip final frame, metrics reported after trial.run()

    try:
        ray.tune.report({
            BASE_SCHEDULER_SPEC["time_attr"]: frame,
            BASE_SCHEDULER_SPEC["metric"]: mt.total_reward_ma
        })
    except Exception as e:
        logger.warning(f"Failed to report metrics: {e}")


def run_ray_search(spec: dict) -> dict:
    """Ray Tune search using Tuner API with Optuna integration and optional ASHA scheduling."""
    name, num_trials = spec["name"], spec["meta"]["max_trial"]
    use_scheduler = spec["meta"].get("search_scheduler") is not None

    # ASHA scheduler requires single-session trials for periodic metric reporting
    # Automatically override max_session to 1 when search_scheduler is specified
    if use_scheduler and spec["meta"]["max_session"] > 1:
        original_max_session = spec["meta"]["max_session"]
        spec["meta"]["max_session"] = 1
        logger.info(
            f"ASHA scheduler enabled: overriding max_session from {original_max_session} to 1 "
            f"(ASHA requires single-session trials for early termination)"
        )

    max_session = spec["meta"]["max_session"]
    metric, mode = ps.at(BASE_SCHEDULER_SPEC, "metric", "mode")

    # Configure scheduler only for single-session with search_scheduler specified
    scheduler = None
    if use_scheduler:
        scheduler_spec = {
            **BASE_SCHEDULER_SPEC,
            "max_t": spec["env"]["max_frame"],
            **spec["meta"]["search_scheduler"],
        }
        scheduler = AsyncHyperBandScheduler(**scheduler_spec)
        logger.info(f"Single-session search with ASHA early termination (grace_period={scheduler_spec.get('grace_period', 'default')} frames)")
    elif max_session > 1:
        logger.info(f"Multi-session search (max_session={max_session}): trials run to completion (no early termination)")
    else:
        logger.info(f"Single-session search without scheduler: trials run to completion")

    logger.info(
        f"Ray Tune search: {num_trials} trials, {metric} ({mode}) | Dashboard: http://127.0.0.1:8265 | Stop: slm-lab --stop-ray"
    )
    tuner = tune.Tuner(
        tune.with_resources(build_run_trial(spec), get_trial_resources(spec)),
        param_space=build_param_space(spec),
        tune_config=tune.TuneConfig(
            search_alg=OptunaSearch(metric=metric, mode=mode),
            scheduler=scheduler,  # None for multi-session or when search_scheduler not specified
            num_samples=num_trials,
        ),
        run_config=tune.RunConfig(
            name="ray_tune",
            storage_path=os.path.join(ROOT_DIR, spec["meta"]["predir"]),
        ),
    )

    results = tuner.fit()
    return extract_trial_results(results, spec)


def cleanup_trial_models(spec, experiment_df, keep_top_n=3):
    """
    Keep only top N trial models after search to reduce disk usage.
    Deletes model files for trials not in top N by performance.
    """
    import glob
    import os
    from slm_lab.lib import util

    # Sort trials by total_reward_ma descending
    sorted_df = experiment_df.sort_values('total_reward_ma', ascending=False)
    keep_trials = set(sorted_df.head(keep_top_n)['trial'].tolist())
    all_trials = set(experiment_df['trial'].tolist())
    remove_trials = all_trials - keep_trials

    if not remove_trials:
        logger.info('No trials to remove - all trials within keep limit')
        return

    # Get model directory from spec's predir (data/{spec_name}_{ts}/model/)
    predir = util.get_predir(spec)
    model_dir = f'{predir}/model'

    if not os.path.exists(model_dir):
        logger.info('Model directory does not exist, skipping cleanup')
        return
    
    # Remove model files for trials not in top N
    # Pattern matches both _tX_*.pt and _tX_sY_*.pt formats
    removed_count = 0
    for trial_idx in remove_trials:
        # Match files with _tX_ pattern (covers both _tX_sY_ and _tX_ckpt-)
        pattern = f'{model_dir}/*_t{trial_idx}_*.pt'
        files = glob.glob(pattern)
        for f in files:
            os.remove(f)
            removed_count += 1

    if removed_count > 0:
        logger.info(f'Cleaned up {removed_count} model files, kept top {keep_top_n} trials: {sorted(keep_trials)}')
