"""String constants for tree comparison plots and examples.

This module provides the StringConstants class with markers, line styles,
labels, titles, filenames, and other string constants used consistently
across all visualization and example functions.
"""

from typing import Literal


class StringConstants:
    """String constants for tree comparison plots and examples.

    This class contains markers, line styles, labels, titles, filenames,
    and other string constants used across the project for consistent
    string values.

    Attributes:
        MARKER_CIRCLE: Circle marker string.
        MARKER_SQUARE: Square marker string.
        MARKER_X: X marker string.
        LINE_DASHED: Dashed line style string.
        LINE_DOTTED: Dotted line style string.
        LINE_SOLID_GREEN: Solid green line style string.
        DEFAULT_OUTPUT_DIR: Default output directory path.
        LABEL_ORIGINAL_CORRECT: Label for original correct points.
        LABEL_SIMPLIFIED_CORRECT: Label for simplified correct points.
        LABEL_BOTH_AGREE_CORRECT: Label for both agree correct points.
        LABEL_BOTH_AGREE_WRONG: Label for both agree wrong points.
        LABEL_BOTH_WRONG_DIFF: Label for both wrong different points.
        LABEL_ORIGINAL_BETTER: Label for original better points.
        LABEL_SIMPLIFIED_BETTER: Label for simplified better points.
        LABEL_EQUAL: Label for equal performance points.
        LABEL_PERFECT_AGREEMENT: Label for perfect agreement.
        LABEL_EQUAL_LOSS: Label for equal loss.
        LABEL_LOSS_CURVE: Label for loss curve.
        LABEL_PREDICTED_PROBABILITY: Label for predicted probability axis.
        LABEL_LOG_LOSS_AXIS: Label for log loss axis.
        LABEL_FALSE_POSITIVE_RATE: Label for false positive rate axis.
        LABEL_TRUE_POSITIVE_RATE: Label for true positive rate axis.
        EDGE_COLOR_BLACK: Black edge color string.
        EDGE_COLOR_GRAY: Gray edge color string.
        PRINT_SEPARATOR: Separator for print output.
        NEWLINE_PREFIX: Newline prefix string.
        FILE_DECISION_TREE_COMPARISON: Filename for decision tree comparison.
        FILE_DECISION_TREE_PREDICTION_SCATTER: Filename for decision tree prediction scatter.
        FILE_DECISION_TREE_LOG_LOSS_SCATTER: Filename for decision tree log loss scatter.
        FILE_DECISION_TREE_LOG_LOSS_CURVE: Filename for decision tree log loss curve.
        FILE_DECISION_TREE_CONFUSION_MATRIX: Filename for decision tree confusion matrix.
        FILE_DECISION_TREE_ROC_CURVES: Filename for decision tree ROC curves.
        TITLE_CONFUSION_MATRICES_TREE: Title for decision tree confusion matrices.
        TITLE_ROC_CURVES_TREE: Title for decision tree ROC curves.
        TITLE_LOG_LOSS_CURVE_TREE: Title for decision tree log loss curve.
        TITLE_DECISION_TREE_SIMPLIFICATION: Title for decision tree simplification.
        LABEL_ORIGINAL_TREE_PREDICTION: Label for original tree prediction.
        LABEL_SIMPLIFIED_TREE_PREDICTION: Label for simplified tree prediction.
        LABEL_ORIGINAL_TREE_LOG_LOSS: Label for original tree log loss.
        LABEL_SIMPLIFIED_TREE_LOG_LOSS: Label for simplified tree log loss.
        DATASET_IRIS: Dataset name for iris.
        DATASET_WINE: Dataset name for wine.
        DATASET_BREAST_CANCER: Dataset name for breast cancer.
        DATASET_DIGITS: Dataset name for digits.
        DATASET_TITANIC: Dataset name for titanic.
        DEFAULT_DATASET: Default dataset name.
    """

    # Dataset constants
    DATASET_IRIS: str = "iris"
    DATASET_WINE: str = "wine"
    DATASET_BREAST_CANCER: str = "breast_cancer"
    DATASET_DIGITS: str = "digits"
    DATASET_TITANIC: str = "titanic"
    DEFAULT_DATASET: str = "iris"

    # Marker constants
    MARKER_CIRCLE: str = "o"
    MARKER_SQUARE: str = "s"
    MARKER_X: str = "x"

    # Line style constants
    LINE_DASHED: str = "k--"
    LINE_DOTTED: str = "k:"
    LINE_SOLID_GREEN: str = "g-"

    # Default output directory
    DEFAULT_OUTPUT_DIR: str = "examples"

    # Agreement / performance labels
    LABEL_ORIGINAL_CORRECT: str = "Original correct"
    LABEL_SIMPLIFIED_CORRECT: str = "Simplified correct"
    LABEL_BOTH_AGREE_CORRECT: str = "Both agree (correct)"
    LABEL_BOTH_AGREE_WRONG: str = "Both agree (wrong)"
    LABEL_BOTH_WRONG_DIFF: str = "Both wrong (diff)"
    LABEL_ORIGINAL_BETTER: str = "Original better"
    LABEL_SIMPLIFIED_BETTER: str = "Simplified better"
    LABEL_EQUAL: str = "Equal"
    LABEL_PERFECT_AGREEMENT: str = "Perfect agreement"
    LABEL_EQUAL_LOSS: str = "Equal loss"
    LABEL_LOSS_CURVE: str = "Loss curve"
    LABEL_PREDICTED_PROBABILITY: str = "Predicted Probability"
    LABEL_LOG_LOSS_AXIS: str = "Log Loss"
    LABEL_FALSE_POSITIVE_RATE: str = "False Positive Rate"
    LABEL_TRUE_POSITIVE_RATE: str = "True Positive Rate"

    # Edge color constants
    EDGE_COLOR_BLACK: str = "black"
    EDGE_COLOR_GRAY: str = "gray"

    # Print formatting constants
    PRINT_SEPARATOR: str = "-"
    NEWLINE_PREFIX: str = "\n"

    # Filename constants for decision tree
    FILE_DECISION_TREE_COMPARISON: str = "decision_tree_comparison.png"
    FILE_DECISION_TREE_PREDICTION_SCATTER: str = "decision_tree_prediction_scatter.png"
    FILE_DECISION_TREE_LOG_LOSS_SCATTER: str = "decision_tree_log_loss_scatter.png"
    FILE_DECISION_TREE_LOG_LOSS_CURVE: str = "decision_tree_log_loss_curve.png"
    FILE_DECISION_TREE_CONFUSION_MATRIX: str = "decision_tree_confusion_matrix.png"
    FILE_DECISION_TREE_ROC_CURVES: str = "decision_tree_roc_curves.png"

    # Title constants for decision tree
    TITLE_CONFUSION_MATRICES_TREE: str = "Confusion Matrices: Original vs Simplified Decision Tree"
    TITLE_ROC_CURVES_TREE: str = "ROC Curves: Original vs Simplified Decision Tree"
    TITLE_LOG_LOSS_CURVE_TREE: str = "Log Loss Curve: Theoretical vs Actual Predictions"
    TITLE_DECISION_TREE_SIMPLIFICATION: str = "Decision Tree Simplification using TreE-Graph"

    # Label constants for decision tree
    LABEL_ORIGINAL_TREE_PREDICTION: str = "Original Tree Prediction"
    LABEL_SIMPLIFIED_TREE_PREDICTION: str = "Simplified Tree Prediction"
    LABEL_ORIGINAL_TREE_LOG_LOSS: str = "Original Tree Log Loss (per sample)"
    LABEL_SIMPLIFIED_TREE_LOG_LOSS: str = "Simplified Tree Log Loss (per sample)"

    # Rule set literals
    RULE_SET_BASIC: Literal["basic"] = "basic"
    RULE_SET_FULL: Literal["full"] = "full"
