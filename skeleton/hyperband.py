"""Optional Hyperband interface for random forests.

Implement the method, connect a suitable package, or replace this interface.
Choose and justify the allocation schedule and how you retain search results.
"""

from __future__ import annotations

from typing import Any

from random_forest import Config, Evaluator, sample_configuration

import numpy as np


def optimise_hyperband(
    evaluator: Evaluator,
    min_trees: int,
    max_trees: int,
    seed: int,
    reduction_factor: int = 3,
) -> tuple[Config, Any]:
    """TODO: implement or configure multiple successive-halving brackets.

    IMPORTANT: this function currently only maximizes

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
    assert eta > 1, "Reduction factor must be greater than 1"
    s_max = int(np.floor(np.log(max_trees/min_trees) / np.log(eta)))
    B = (s_max+1) * max_trees

    best_score = float('-inf')
    best_config = None
    history = []

    for s in range(s_max, -1, -1):
        n = int(np.ceil((B/max_trees)*(eta**s / (s+1))))
        r = int(max_trees*(eta**-(s_max-s)))
        configs = [(sample_configuration(rng), None) for _ in range(n)]
        
        for i in range(s+1):
            r_i = int(r*(eta**i))
            L = [evaluator(config, r_i, seed, model) for config, model in configs]
            history.extend([{**{k: v for k, v in result.items() if k != "model"},"n_trees":r_i} for result in L])
            scores = [l["objective"] for l in L]
            max_idx = np.argmax(scores)
            best_current_score = scores[max_idx]
            if best_current_score > best_score:
                best_score = best_current_score
                best_config = configs[max_idx][0]

            if i < s:
                keep = max(1, int(np.floor(len(configs) / eta)))
                keep_idx = np.argsort(scores)[-keep:]
                configs = [(configs[idx][0], L[idx]["model"]) for idx in keep_idx]
          
    return best_config, history
        