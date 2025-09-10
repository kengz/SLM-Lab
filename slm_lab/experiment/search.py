import os
from copy import deepcopy

import pydash as ps
import ray
import ray.tune as tune
import torch
from ray.tune.search.optuna import OptunaSearch

from slm_lab import ROOT_DIR
from slm_lab.experiment.analysis import METRICS_COLS
from slm_lab.lib import logger, util

logger = logger.get_logger(__name__)


def build_param_space(spec):
    """
    Build Ray Tune parameter space from SLM-Lab spec.

    Specify a config space in spec using `"{key}__{space_type}": [args]` format.
    Where `{space_type}` is any Ray Tune search space function from:
    https://docs.ray.io/en/latest/tune/api/search_space.html

    Two argument patterns:
    - Most functions: use `[arg1, arg2, ...]` → `tune.func(*args)`
    - choice: use `[item1, item2, ...]` → `tune.choice([items])`

    Examples:
    - `"gamma__uniform": [0.95, 0.999]` → `tune.uniform(0.95, 0.999)`
    - `"lr__loguniform": [0.0001, 0.01]` → `tune.loguniform(0.0001, 0.01)`
    - `"temperature__randn": [1.0, 0.1]` → `tune.randn(1.0, 0.1)`
    - `"batch_size__choice": [16, 32, 64]` → `tune.choice([16, 32, 64])`
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


def inject_config(spec, config):
    """Inject flattened config into SLM Lab spec."""
    spec = deepcopy(spec)
    spec.pop("search", None)

    # Get trial index from Ray Tune trial dir, e.g. /tmp/ray/.../run_trial_da41e96b_1_...
    trial_dir = ray.tune.get_context().get_trial_dir()
    trial_name = ray.tune.get_context().get_trial_name()
    trial_folder = trial_dir.split("/")[-1]
    trial_index = int(trial_folder.replace(f"{trial_name}_", "").split("_")[0]) - 1
    ps.set_(spec, "meta.trial", trial_index)

    for k, v in config.items():
        ps.set_(spec, k, v)
    return spec


def build_run_trial(base_spec):
    """Create a Ray Tune trial runner with the base spec."""

    def run_trial(config):
        from slm_lab.experiment.control import Trial

        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        spec = inject_config(base_spec, config)
        scalar_metrics = Trial(spec).run()
        tune.report(metrics=scalar_metrics)
        return scalar_metrics

    return run_trial


def get_trial_resources(spec):
    """Build resource configuration for Ray Tune trials."""
    max_session = spec["meta"]["max_session"]
    return {"cpu": max_session, "gpu": max_session if torch.cuda.is_available() else 0}


def extract_trial_results(results):
    """Extract trial results into SLM-Lab format."""
    trial_results = {}
    for i, result in enumerate(results):
        metrics_data = ps.pick(result.metrics, *METRICS_COLS)
        trial_results[i] = {**result.config, **metrics_data}
    return trial_results


def run_ray_search(spec):
    """Ray Tune search using Tuner API with Optuna integration."""
    name, num_trials = spec["name"], spec["meta"]["max_trial"]
    logger.info(f"Running Ray Tune for spec {name} with {num_trials} trials")
    logger.info("Note: use 'slm-lab --kill-ray' to stop (Ray SIGTERM issue)")

    tuner = tune.Tuner(
        tune.with_resources(build_run_trial(spec), get_trial_resources(spec)),
        param_space=build_param_space(spec),
        tune_config=tune.TuneConfig(
            metric="final_return_ma",
            mode="max",
            search_alg=OptunaSearch(),
            num_samples=num_trials,
        ),
        run_config=tune.RunConfig(
            name=name,
            storage_path=os.path.join(ROOT_DIR, "data", "ray_tune"),
        ),
    )

    results = tuner.fit()
    return extract_trial_results(results)
