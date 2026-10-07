"""Optional Hyperband interface for random forests.

Implement the method, connect a suitable package, or replace this interface.
Choose and justify the allocation schedule and how you retain search results.
"""

from __future__ import annotations

from typing import Any

from random_forest import Config, Evaluator, sample_configuration

from optuna.pruners import HyperbandPruner
from optuna.integration import OptunaSearchCV

from sklearn.model_selection import HalvingRandomSearchCV, TimeSeriesSplit
import numpy as np


TIMESERIES_CV = TimeSeriesSplit() 

def optimise_hyperband(
    evaluator: Evaluator,
    min_trees: int,
    max_trees: int,
    seed: int,
    reduction_factor: int = 3,
) -> tuple[Config, Any]:
    """TODO: implement or configure multiple successive-halving brackets.

    Use the shared search space. Start brackets with different numbers of
    configurations and trees per forest, between min_trees and max_trees.
    At each stage, keep the better configurations and give them more trees,
    keeping other settings fixed. Use reduction_factor for the decrease in
    configuration count and increase in trees.

    Compare validation objectives consistently: respect whether higher or lower
    values are better. Explain your schedule, refitting or warm starts, and
    how validation results determine the final selection.
    Return the selected configuration and results needed for your analysis.
    """
    rng = np.random.default_rng(seed)

    eta = reduction_factor
    s_max = np.floor(np.log(max_trees))
    B = (s_max+1) * max_trees

    best_so_far = float('-inf')

    all_hyperband_configs = {}
    for s in list(range(s_max+1))[::-1]:
        n = np.ceil((B/max_trees)*(eta**s / (s+1)))
        r = max_trees*(eta**-s)
        configs = [sample_configuration(rng) for _ in range(n)]

        all_hyperband_configs.get(f"{s}", []).append(configs) # logging

        for i in range(s+1):
            n_i = np.floor(n*(eta**-i))
            r_i = r*(eta**i)
            L = [evaluator(config, r_i, seed)["objective"] for config in configs]
            configs = configs[np.argpartition(L, -np.floor(n_i/eta))[-np.floor(n_i/eta):]]

            all_hyperband_configs.get(f"{s}", []).append(configs) # logging
    
    # bracket_configs = [81, 27, 9, 6, 5]

    # param_distributions = None
    # rng = np.random.default_rng(seed)
    # for bracket in bracket_configs:
    #     s = int(np.floor(np.log(max_trees / bracket) / np.log(reduction_factor)))
    #     s = max(0, min(4, s))
    #     r_0 = int(max_trees * (reduction_factor ** (-s)))

    #     configs = [sample_configuration(rng) for _ in range(bracket)]

    #     # 2. Run Successive Halving for this single bracket
    #     for resource_step in range(s + 1):
    #         n_i = len(configs)
    #         r_i = int(r_0 * (reduction_factor**resource_step))

    #         scores = []
    #         for cfg in configs:
    #             est = clone(estimator).set_params(n_estimators=r_i, **cfg)
    #             score = np.mean(cross_val_score(est, X, y, cv=cv))
    #             scores.append(score)

    #             cv_results.append(
    #                 {
    #                     "starting_n": n_configs,
    #                     "resource": r_i,
    #                     "params": cfg,
    #                     "score": score,
    #                 }
    #             )

    #             if score > best_score:
    #                 best_score = score
    #                 best_params = cfg

    #         # Top 1/eta candidates survive to the next round
    #         k = max(1, int(n_i / eta))
    #         top_indices = np.argsort(scores)[-k:]
    #         configs = [configs[idx] for idx in top_indices]

    # return best_params, best_score, cv_results
