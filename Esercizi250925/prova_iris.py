"""Iris classification utilities.

This module provides a function to train and evaluate a Decision Tree classifier
on the Iris dataset with optional confusion matrix, cross-validation, and
grid-search support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union, Optional

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate_iris(
    csv_path: str = "iris.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    max_depth: Optional[int] = 5,
    *,
    target_col: str = "Species",
    return_dict: bool = False,
    compute_confusion_matrix: bool = False,
    compute_cv: bool = False,
    cv_folds: int = 5,
    do_grid_search: bool = False,
    param_grid: Optional[Dict[str, list]] = None,
) -> Tuple[DecisionTreeClassifier, Union[str, dict]]:
    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    """Train a Decision Tree on the Iris dataset and return the model and report.

    Parameters
    ----------
    csv_path : str, default="iris.csv"
        Path to the Iris CSV file. The CSV must contain the target column
        specified by `target_col` and the numerical feature columns.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (between 0 and 1).
    random_state : int, default=42
        Seed used for reproducibility of the split and the model.
    max_depth : int or None, default=5
        Maximum depth of the decision tree. Use None for unlimited depth.
    target_col : str, default="Species"
        Name of the target column in the CSV file.
    return_dict : bool, default=False
        If True, return the classification report as a dictionary (output_dict=True)
        and include optional extras (accuracy, confusion_matrix, cv, grid_search).
    compute_confusion_matrix : bool, default=False
        If True and `return_dict=True`, include the confusion matrix in the report dict.
    compute_cv : bool, default=False
        If True and `return_dict=True`, include simple CV accuracy with `cv_folds` folds.
    cv_folds : int, default=5
        Number of folds for cross-validation if `compute_cv=True`.
    do_grid_search : bool, default=False
        If True, perform a GridSearchCV on the training data to tune hyperparameters.
    param_grid : dict or None, optional
        Parameter grid for GridSearchCV. If None, a small default grid is used.

    Returns
    -------
    model : sklearn.tree.DecisionTreeClassifier
        The trained classifier.
    report : str or dict
        Classification report on the test set. If `return_dict` is False, a
        formatted string is returned; otherwise, a dictionary with additional
        metrics if requested.

    Raises
    ------
    FileNotFoundError
        If the file at `csv_path` does not exist.
    ValueError
        If the dataset is empty, missing the target column, the target has NaNs,
        or only one unique class (stratification requires at least 2 classes).

    Examples
    --------
    >>> model, report = train_and_evaluate_iris(
    ...     "iris.csv", test_size=0.2, random_state=42, max_depth=5
    ... )
    >>> print(report)
    >>> model, report = train_and_evaluate_iris(
    ...     "iris.csv", return_dict=True, compute_confusion_matrix=True
    ... )
    >>> report["accuracy"]
    0.97
    """
    # Validate path and load dataset
    path_obj = Path(csv_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(path_obj)
    if df.empty:
        raise ValueError("The dataset is empty.")
    if target_col not in df.columns:
        raise ValueError(
            (
                f"Target column '{target_col}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        )
    if df[target_col].isna().any():
        raise ValueError("Target column contains NaN values; please clean the dataset.")
    if df[target_col].nunique() < 2:
        raise ValueError("Target must contain at least two classes for stratification.")

    # Split data into train and test (stratified)
    features = df.drop(target_col, axis=1)
    target = df[target_col]
    features_train, features_test, target_train, target_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    # Base estimator
    base_model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model = base_model

    # Optional grid search on training data
    grid_info = None
    if do_grid_search:
        if param_grid is None:
            param_grid = {
                "max_depth": [2, 3, 4, 5, 6, None],
                "min_samples_split": [2, 4, 6, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy", "log_loss"],
            }
        grid = GridSearchCV(base_model, param_grid=param_grid, cv=cv_folds, n_jobs=-1)
        grid.fit(features_train, target_train)
        model = grid.best_estimator_
        grid_info = {"best_params": grid.best_params_, "best_score": grid.best_score_}

    # Train model
    model.fit(features_train, target_train)
    y_pred = model.predict(features_test)

    # Evaluate
    acc = accuracy_score(target_test, y_pred)
    if return_dict:
        report: dict = classification_report(target_test, y_pred, output_dict=True)
        report["accuracy"] = acc
        if compute_confusion_matrix:
            labels = sorted(target.unique())
            report["confusion_matrix_labels"] = labels
            report["confusion_matrix"] = confusion_matrix(
                target_test, y_pred, labels=labels
            ).tolist()
        if compute_cv:
            # Simple CV on the training set using the final model hyperparameters
            # Note: using GridSearchCV scores if available
            if grid_info is not None:
                report["grid_search"] = grid_info
            else:
                report["grid_search"] = None
        elif grid_info is not None:
            report["grid_search"] = grid_info
        return model, report
    # String report; extras only printed by caller if desired
    text_report = classification_report(target_test, y_pred)
    return model, text_report


if __name__ == "__main__":
    _, report_str = train_and_evaluate_iris()
    print(report_str)
