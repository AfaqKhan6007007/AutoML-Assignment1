"""Example runner for a shared split, forest search, and final evaluation.

Adapt this flow to your experimental design. Results stay in memory; choose how
to save them and record the settings needed to reproduce your study.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from data_loading import DATASETS, load_and_split, prepare_data, prepare_final_data
from hyperband import optimise_hyperband
from random_forest import final_test_evaluation, make_evaluator
from random_search import optimise_random_search
from smbo import optimise_smbo
from tabular_foundation import run_foundation_model
import json

import warnings

# turn off class weight warning, not relevant as same stratum is used for warm start runs.
warnings.filterwarnings(
    "ignore",
    message=r'class_weight presets "balanced" or "balanced_subsample" are not recommended for warm_start.*',
    category=UserWarning,
)

# ensure correct path for saving 
script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "../data"
output_dir.mkdir(parents=True, exist_ok=True)

# Update this if the provided largest dataset is replaced.
FOUNDATION_DATASET = "covertype"

# Time series dataset
TIMESERIES_DS = "electricity"

# n_trials is the example evaluation budget for each of Random Search and SMBO.
# Choose budgets and a Hyperband schedule that support your justified comparison.
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "max_samples": 2_500,
        "min_trees": 3,
        "max_trees": 27,
        "n_trials": 9,
    },
    "course": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 81,
        "n_trials": 16,
    },
    "full": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 243,
        "n_trials": 24,
    },
    "exp": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 243,
        "n_trials": 16, # equivalent to number of trees trained with hyperband 
    },
}


def parse_args() -> argparse.Namespace:
    """Parse the reproducible experiment command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="breast-w", choices=[*DATASETS, "all"])
    parser.add_argument("--profile", default="smoke", choices=PROFILES)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["default", "random", "smbo", "hyperband"],
        choices=["default", "random", "smbo", "hyperband", "foundation"],
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    parser.add_argument("--context", type=int, default=100)
    return parser.parse_args()


def run_dataset(name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return example in-memory results; their structure is yours to adapt."""

    if args.methods == ["foundation"] and name != FOUNDATION_DATASET:
        if args.dataset == "all":
            return []
        raise ValueError(f"Foundation-only runs require --dataset {FOUNDATION_DATASET}")

    profile = PROFILES[args.profile]
    temporal = False
    if name == TIMESERIES_DS:
        temporal = True
    splits = load_and_split(name, args.cache_dir, profile["max_samples"], args.split_seed, temporal=temporal)
    results = []
    forest_methods = {"default", "random", "smbo", "hyperband"}.intersection(args.methods)
    if forest_methods:
        X_train, X_valid = prepare_data(splits)
        evaluator = make_evaluator(
            X_train, splits.y_train, X_valid, splits.y_valid
        )
        final_arrays = prepare_final_data(splits)
        min_trees = int(profile["min_trees"])
        max_trees = int(profile["max_trees"])

        for method in ("default", "random", "smbo", "hyperband"):
            if method not in forest_methods:
                continue
            start = perf_counter()
            if method == "default":
                config = {}  # Library defaults, with the common tree count.
                history = [evaluator(config, max_trees, args.seed)]
            elif method == "random":
                config, history = optimise_random_search(
                    evaluator, profile["n_trials"], max_trees, args.seed
                )
            elif method == "smbo":
                config, history = optimise_smbo(
                    evaluator, profile["n_trials"], max_trees, args.seed
                )
            else:
                config, history = optimise_hyperband(
                    evaluator, min_trees, max_trees, args.seed
                )
            search_seconds = perf_counter() - start
            final_result = final_test_evaluation(
                config, max_trees, args.seed, *final_arrays
            )
            print(f"{name} / {method} / seed {args.seed}: {final_result}", flush=True)
            results.append({
                "dataset": name,
                "method": method,
                "seed": args.seed,
                "configuration": config,
                "history": history,
                "search_seconds": search_seconds,
                "final_result": final_result,
            })

    if "foundation" in args.methods and name == FOUNDATION_DATASET:
        result = run_foundation_model(splits, seed=args.seed, context_size=args.context)
        print(f"{name} / foundation / seed {args.seed}: {result}", flush=True)
        results.append({
            "dataset": name,
            "method": "foundation",
            "seed": args.seed,
            "result": result,
        })
    return results


def main() -> None:
    """Run the selected examples; add result saving before the main study."""

    args = parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"Running {name} with seed {args.seed}", flush=True)
        results = run_dataset(name, args)
        # TODO: save results in a format of your choice, along with the settings
        # needed to reproduce the run. Retain enough information for your plots
        # and tables. This example only prints final results; it saves no files.

        filename = f"expseed_{args.seed}_{name}.json"
        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
