"""Optional SMBO interface for the shared forest search space.

Implement the method, connect a suitable package, or replace this interface.
Choose and explain the method's settings and how you retain search results.
"""

from __future__ import annotations

from typing import Any

from random_forest import Config, Evaluator, SEARCH_SPACE

import optuna

def optimise_smbo(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """TODO: use SMBO to choose configurations based on previous evaluations.

    Current implementation always maximizes. 

    Use the shared search space and train each forest with n_trees trees.
    Use up to n_trials evaluations, including any initial evaluations.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """

    history = []
    def objective(trial):
        config = {hp: trial.suggest_categorical(hp, choices) for hp, choices in SEARCH_SPACE.items()}
    
        result = evaluator(config, n_trees, seed)
        history.append(result)
        return result["objective"]
                     
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    return Config(**study.best_params), history

