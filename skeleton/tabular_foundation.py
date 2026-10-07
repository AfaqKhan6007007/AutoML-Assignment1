"""Optional interface for a pre-trained model on the largest selected dataset.

Use a suitable package directly, adapt this interface, or organise your own experiment.
"""

from __future__ import annotations

from typing import Any

from data_loading import DataSplits, prepare_final_data

from tabpfn import TabPFNClassifier

def run_foundation_model(splits: DataSplits, seed: int) -> Any:
    """TODO: evaluate a pre-trained tabular foundation model.

    Choose and justify the model, how you use it, and the data available to it.
    Evaluate on the same complete test set as the forests, keeping it separate
    from training, adaptation, and model selection.
    Retain results for comparing performance and compute with the baseline
    and each tuned forest. See Section 3.5 of the assignment.
    """
    X_fit, y_train_valid, X_test, y_test = prepare_final_data(splits)
    X_context, _, y_context, _ = train_test_split(
    X_fit,
    y_train_valid,
    train_size=1000,
    stratify=y_train_valid,
    random_state=seed,  
    )
    clf = TabPFNClassifier()
    
    clf.fit(X_context, y_context)
    predictions = clf.predict(X_test)


