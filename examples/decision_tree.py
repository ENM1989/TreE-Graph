#!/usr/bin/env python3
"""Simplification example for TreE-Graph supporting multiple scikit-learn datasets.

This example demonstrates the workflow for simplifying scikit-learn decision
trees using e-graph rewriting across various datasets (e.g. iris, wine,
breast_cancer, digits).
Includes hyperparameter tuning with GridSearchCV and k-fold cross-validation.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Add project root to sys.path so the script can be run directly
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from matplotlib import use as matplotlib_use

from examples.constants import Constants

matplotlib_use(Constants.MATPLOTLIB_BACKEND_AGG)
# Non-interactive backend for headless environments

from numpy import array_equal
from numpy import bincount
from numpy import mean
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from examples.dataset_loader import DatasetInput
from examples.dataset_loader import get_supported_datasets
from examples.dataset_loader import load_dataset
from examples.metrics_calculator import MetricsCalculator
from examples.tree_visualizer import TreeVisualizer
from tree_graph import TreeSimplifier


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the example script.

    Args:
        args: Optional list of argument strings. Defaults to sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    supported = ", ".join(get_supported_datasets())
    parser = argparse.ArgumentParser(
        description="Run decision tree simplification example on scikit-learn datasets.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=Constants.DEFAULT_DATASET,
        help=(
            f"Scikit-learn dataset to use (default: {Constants.DEFAULT_DATASET}). "
            f"Options: {supported}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=Constants.DEFAULT_OUTPUT_DIR,
        help=f"Output directory for plots (default: {Constants.DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for saved plot filenames (defaults to '<dataset>_' when dataset is not iris).",
    )
    return parser.parse_args(args)


def main(
    dataset: DatasetInput = Constants.DEFAULT_DATASET,
    output_dir: str = Constants.DEFAULT_OUTPUT_DIR,
    output_prefix: str | None = None,
    args: list[str] | None = None,
) -> None:
    """Run the simplification example with a selectable scikit-learn dataset.

    This function:
    1. Loads the specified dataset (e.g. iris, wine, breast_cancer, digits)
    2. Performs hyperparameter tuning with cross-validation
    3. Trains a decision tree with best parameters
    4. Simplifies the tree using e-graph rewriting
    5. Compares original and simplified trees
    6. Generates visualizations and metrics

    Args:
        dataset: Dataset name or data object (default: 'iris').
        output_dir: Directory path for saving plots (default: 'examples').
        output_prefix: Optional prefix for output plot filenames.
        args: Optional command line argument list to override parameters.
    """
    if args is not None:
        parsed_args = parse_args(args)
        dataset = parsed_args.dataset
        output_dir = parsed_args.output_dir
        output_prefix = parsed_args.output_prefix

    # Load the requested dataset
    features, labels, feature_names, class_names = load_dataset(dataset)

    dataset_name = dataset if isinstance(dataset, str) else type(dataset).__name__
    print(f"Dataset: {dataset_name}")
    print(f"  - Samples: {features.shape[Constants.ZERO]}")
    print(f"  - Features: {features.shape[Constants.ONE]}")
    print(f"  - Classes: {len(class_names)} ({', '.join(class_names[: Constants.TEN])})")
    print()

    # Stratified split when possible
    label_counts = bincount(labels)
    min_class_count = (
        int(min(label_counts)) if len(label_counts) > Constants.ZERO else Constants.ZERO
    )
    stratify = labels if min_class_count >= Constants.TWO else None

    # Split data into train and test sets
    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        labels,
        test_size=Constants.TEST_SIZE,
        random_state=Constants.RANDOM_STATE,
        stratify=stratify,
    )

    # Define hyperparameter grid for tuning
    param_grid: dict[str, list[Any]] = {
        "max_depth": Constants.GRID_MAX_DEPTH_VALUES,
        "min_samples_split": Constants.GRID_MIN_SAMPLES_SPLIT_VALUES,
        "min_samples_leaf": Constants.GRID_MIN_SAMPLES_LEAF_VALUES,
    }

    # Set up k-fold cross-validation
    train_counts = bincount(labels_train)
    min_train_count = (
        int(min(train_counts)) if len(train_counts) > Constants.ZERO else Constants.ZERO
    )
    k_folds = (
        min(Constants.K_FOLDS, min_train_count)
        if min_train_count >= Constants.TWO
        else Constants.K_FOLDS
    )
    cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=Constants.RANDOM_STATE)

    # Create grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=Constants.RANDOM_STATE),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=Constants.MINUS_ONE,
        verbose=Constants.ONE,
    )

    print("Performing GridSearchCV with k-fold cross-validation...")
    print(f"  - {k_folds}-fold cross-validation")
    print(f"  - Parameter grid: {param_grid}")
    print()

    # Fit grid search to find best hyperparameters
    grid_search.fit(features_train, labels_train)

    # Get the best estimator
    clf = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best parameters: {best_params}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    # Get original tree statistics
    original_nodes = clf.tree_.node_count
    print(f"\nOriginal tree: {original_nodes} nodes")

    # Simplify the tree using e-graph rewriting
    simplifier = TreeSimplifier(max_iterations=Constants.MAX_ITERATIONS)
    simplified_clf = simplifier.simplify(clf, features_train, labels_train)

    # Get simplified tree statistics
    simplified_nodes = simplified_clf.tree_.node_count
    print(f"Simplified tree: {simplified_nodes} nodes")

    # Calculate reduction percentage
    reduction = (Constants.ONE - simplified_nodes / original_nodes) * Constants.HUNDRED
    print(f"Reduction: {reduction:.1f}%")

    # Verify predictions are preserved on training data
    orig_pred_train = clf.predict(features_train)
    simp_pred_train = simplified_clf.predict(features_train)
    predictions_match = array_equal(orig_pred_train, simp_pred_train)
    print(f"Training predictions preserved: {predictions_match}")

    # Compare accuracy on test data
    orig_accuracy = clf.score(features_test, labels_test)
    simp_accuracy = simplified_clf.score(features_test, labels_test)

    print(f"\nTest accuracy (original): {orig_accuracy:.2%}")
    print(f"Test accuracy (simplified): {simp_accuracy:.2%}")

    # Initialize calculator and visualizer
    calculator = MetricsCalculator()
    visualizer = TreeVisualizer(output_dir=output_dir)

    # Determine filename prefix for saved visualizations
    prefix = ""
    if output_prefix is not None:
        prefix = output_prefix
    elif (
        isinstance(dataset, str)
        and dataset.strip().lower().replace("-", "_") != Constants.DATASET_IRIS
    ):
        prefix = f"{dataset.strip().lower().replace('-', '_')}_"

    comparison_path = (
        f"{output_dir}/{prefix}{Constants.FILE_DECISION_TREE_COMPARISON}" if prefix else None
    )
    pred_scatter_path = (
        f"{output_dir}/{prefix}{Constants.FILE_DECISION_TREE_PREDICTION_SCATTER}"
        if prefix
        else None
    )
    log_loss_scatter_path = (
        f"{output_dir}/{prefix}{Constants.FILE_DECISION_TREE_LOG_LOSS_SCATTER}" if prefix else None
    )
    log_loss_curve_path = (
        f"{output_dir}/{prefix}{Constants.FILE_DECISION_TREE_LOG_LOSS_CURVE}" if prefix else None
    )
    confusion_path = (
        f"{output_dir}/{prefix}{Constants.FILE_DECISION_TREE_CONFUSION_MATRIX}" if prefix else None
    )
    roc_path = f"{output_dir}/{prefix}{Constants.FILE_DECISION_TREE_ROC_CURVES}" if prefix else None

    # Get test predictions
    orig_pred = clf.predict(features_test)
    simp_pred = simplified_clf.predict(features_test)
    orig_proba = clf.predict_proba(features_test)
    simp_proba = simplified_clf.predict_proba(features_test)

    # Compute all metrics
    orig_metrics = calculator.compute_all_metrics(labels_test, orig_pred, orig_proba, class_names)
    simp_metrics = calculator.compute_all_metrics(labels_test, simp_pred, simp_proba, class_names)

    # Print comparison
    print("\nPrecision (macro avg):")
    print(f"  Original:   {orig_metrics['precision_macro']:.4f}")
    print(f"  Simplified: {simp_metrics['precision_macro']:.4f}")

    print("\nRecall (macro avg):")
    print(f"  Original:   {orig_metrics['recall_macro']:.4f}")
    print(f"  Simplified: {simp_metrics['recall_macro']:.4f}")

    print("\nF1-Score (macro avg):")
    print(f"  Original:   {orig_metrics['f1_macro']:.4f}")
    print(f"  Simplified: {simp_metrics['f1_macro']:.4f}")

    # Print per-class classification report
    labels_list = list(range(len(class_names)))
    print("\nClassification Report (Original):")
    print(
        classification_report(
            labels_test,
            orig_pred,
            labels=labels_list,
            target_names=class_names,
            zero_division=0.0,
        )
    )

    print("Classification Report (Simplified):")
    print(
        classification_report(
            labels_test,
            simp_pred,
            labels=labels_list,
            target_names=class_names,
            zero_division=0.0,
        )
    )

    # Print sensitivity/specificity per class
    print("\nSensitivity and Specificity (per class):")
    header = (
        f"{'Class':<15} {'Original Sens':>15} {'Original Spec':>15} "
        f"{'Simp Sens':>12} {'Simp Spec':>12}"
    )
    print(header)
    print(Constants.PRINT_SEPARATOR * Constants.HEADER_WIDTH_70)
    for class_name in class_names:
        orig_sens = orig_metrics["sensitivity"][class_name]
        orig_spec = orig_metrics["specificity"][class_name]
        simp_sens = simp_metrics["sensitivity"][class_name]
        simp_spec = simp_metrics["specificity"][class_name]
        row = (
            f"{class_name:<15} {orig_sens:>15.4f} {orig_spec:>15.4f} "
            f"{simp_sens:>12.4f} {simp_spec:>12.4f}"
        )
        print(row)

    # Generate and save ROC curves
    print("\nROC AUC Scores (per class):")
    print(f"{'Class':<15} {'Original AUC':>15} {'Simplified AUC':>15}")
    print(Constants.PRINT_SEPARATOR * Constants.HEADER_WIDTH_45)
    roc_result = visualizer.plot_roc_curves(
        clf,
        simplified_clf,
        features_test,
        labels_test,
        class_names,
        save_path=roc_path,
    )
    for class_name in class_names:
        orig_auc = roc_result.auc_scores["original"][class_name]
        simp_auc = roc_result.auc_scores["simplified"][class_name]
        print(f"{class_name:<15} {orig_auc:>15.4f} {simp_auc:>15.4f}")

    print("\nMacro Average AUC:")
    print(f"  Original:   {mean(list(roc_result.auc_scores['original'].values())):.4f}")
    print(f"  Simplified: {mean(list(roc_result.auc_scores['simplified'].values())):.4f}")

    # Compute and print log loss
    print("\nLog Loss (Categorical Cross-Entropy):")
    print(f"  Original:   {orig_metrics['log_loss']:.4f}")
    print(f"  Simplified: {simp_metrics['log_loss']:.4f}")
    print(f"  Difference: {simp_metrics['log_loss'] - orig_metrics['log_loss']:+.4f}")

    # Print per-class cross-entropy
    print("\nCategorical Cross-Entropy (per class):")
    print(f"{'Class':<15} {'Original':>15} {'Simplified':>15} {'Difference':>12}")
    print(Constants.PRINT_SEPARATOR * Constants.HEADER_WIDTH_57)
    for class_name in class_names:
        orig_ce = max(0.0, orig_metrics["per_class_ce"][class_name])
        simp_ce = max(0.0, simp_metrics["per_class_ce"][class_name])
        diff = simp_ce - orig_ce
        print(f"{class_name:<15} {orig_ce:>15.4f} {simp_ce:>15.4f} {diff:>+12.4f}")

    # Generate visualizations
    visualizer.plot_trees(
        clf, simplified_clf, feature_names, class_names, save_path=comparison_path
    )
    visualizer.plot_prediction_scatter(
        clf, simplified_clf, features_test, labels_test, class_names, save_path=pred_scatter_path
    )
    visualizer.plot_log_loss_scatter(
        clf,
        simplified_clf,
        features_test,
        labels_test,
        class_names,
        save_path=log_loss_scatter_path,
    )
    visualizer.plot_log_loss_curve(
        clf,
        simplified_clf,
        features_test,
        labels_test,
        class_names,
        save_path=log_loss_curve_path,
    )
    visualizer.plot_confusion_matrices(
        clf, simplified_clf, features_test, labels_test, class_names, save_path=confusion_path
    )

    # Print cross-validation results for all parameter combinations
    print("\nCross-validation results:")
    cv_results = grid_search.cv_results_
    for idx, (params, mean_score, std_score) in enumerate(
        zip(
            cv_results["params"],
            cv_results["mean_test_score"],
            cv_results["std_test_score"],
            strict=True,
        )
    ):
        print(f"  {idx + 1:2d}. {params} -> {mean_score:.4f} (+/- {std_score:.4f})")

    print("\nDone!")


if __name__ == "__main__":
    cli_args = parse_args(sys.argv[Constants.ONE :])
    main(
        dataset=cli_args.dataset,
        output_dir=cli_args.output_dir,
        output_prefix=cli_args.output_prefix,
    )
