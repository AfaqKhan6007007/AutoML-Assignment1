"""Optional interface for a pre-trained model on the largest selected dataset.

Use a suitable package directly, adapt this interface, or organise your own experiment.
"""

from __future__ import annotations

from typing import Any

from time import perf_counter

from data_loading import DataSplits, prepare_final_data

from sklearn.model_selection import train_test_split

from random_forest import predictive_metrics

from dotenv import load_dotenv
load_dotenv()

from tabpfn import TabPFNClassifier




def run_foundation_model(splits: DataSplits, seed: int, context_size: int=1000) -> Any:
    """TODO: evaluate a pre-trained tabular foundation model.

    Choose and justify the model, how you use it, and the data available to it.
    Evaluate on the same complete test set as the forests, keeping it separate
    from training, adaptation, and model selection.
    Retain results for comparing performance and compute with the baseline
    and each tuned forest. See Section 3.5 of the assignment.
    
    Capped at 50k test rows per call
    """
    # return
    # TODO: adjust so that single predict calls do not exceed 50k rows
    start = perf_counter()
    X_fit, y_train_valid, X_test, y_test = prepare_final_data(splits)
    X_context, _, y_context, _ = train_test_split(
    X_fit,
    y_train_valid,
    train_size=context_size,
    stratify=y_train_valid,
    random_state=seed,  
    )
    clf = TabPFNClassifier(
        random_state=seed, 
        model_path='tabpfn-v2.5-classifier-v2.5_default.ckpt',
        device='cpu',
        n_estimators=2,
        n_preprocessing_jobs=4,
        )
    
    clf.fit(X_context, y_context)
    return {
            "metrics": predictive_metrics(clf, X_test, y_test),
            "elapsed_sec": float(perf_counter() - start),
            }
    
