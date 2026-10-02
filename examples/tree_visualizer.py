"""Visualization for decision tree comparison.

This module provides the TreeVisualizer class for creating visualizations
that compare original and simplified decision trees.
"""

from typing import Any

from matplotlib import pyplot as plt
from matplotlib import use as matplotlib_use
from matplotlib.lines import Line2D
from numpy import any as np_any
from numpy import arange
from numpy import linspace
from numpy import log as np_log
from numpy import max as np_max
from numpy import mean
from numpy import min as np_min
from numpy import ndarray
from numpy import random
from numpy import sum as np_sum
from numpy import zeros
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from examples.constants import Constants

matplotlib_use(
    Constants.MATPLOTLIB_BACKEND_AGG
)  # Non-interactive backend for headless environments

from examples.log_loss_result import LogLossResult
from examples.roc_result import ROCResult


class TreeVisualizer:
    """Visualize and compare original vs simplified decision trees.

    This class provides methods for creating various visualizations that
    help compare the behavior and performance of original and simplified
    decision tree models.

    Attributes:
        output_dir: Directory for saving visualization files.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from tree_graph import TreeSimplifier
        >>> from examples.visualization import TreeVisualizer
        >>> X, y = load_iris(return_X_y=True)
        >>> clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        >>> clf.fit(X, y)
        >>> simplifier = TreeSimplifier()
        >>> simplified = simplifier.simplify(clf, X, y)
        >>> visualizer = TreeVisualizer(output_dir="output")
        >>> visualizer.plot_trees(clf, simplified, ['feature1', 'feature2'], ['class1', 'class2'])
    """

    def __init__(self, output_dir: str = Constants.DEFAULT_OUTPUT_DIR) -> None:
        """Initialize the visualizer.

        Args:
            output_dir: Directory path for saving visualization files.
                       Default is "examples".
        """
        self.output_dir = output_dir

    def _get_save_path(self, filename: str) -> str:
        """Construct full save path for a file.

        Args:
            filename: Name of the file to save.

        Returns:
            Full path including output directory.
        """
        return f"{self.output_dir}/{filename}"

    def _add_jitter(
        self,
        values: ndarray,
        amount: float,
        seed: int = Constants.RANDOM_STATE,
    ) -> ndarray:
        """Add random jitter to values for scatter plot visibility.

        Args:
            values: Array of values to jitter.
            amount: Standard deviation of jitter noise.
            seed: Random seed for reproducibility.

        Returns:
            Array with jitter added.
        """
        rng = random.default_rng(seed)
        return values + rng.normal(Constants.ZERO, amount, len(values))

    def _create_legend_elements(
        self,
        labels: list[str],
        colors: list[str],
        marker: str = Constants.MARKER_CIRCLE,
    ) -> list[Line2D]:
        """Create legend elements for matplotlib plots.

        Args:
            labels: List of legend labels.
            colors: List of marker colors (same length as labels).
            marker: Marker style to use.

        Returns:
            List of Line2D objects for use in ax.legend().
        """
        elements = []
        for label, color in zip(labels, colors, strict=True):
            elements.append(
                Line2D(
                    [Constants.ZERO],
                    [Constants.ZERO],
                    marker=marker,
                    color="w",
                    markerfacecolor=color,
                    markersize=Constants.MARKER_SIZE_10,
                    label=label,
                )
            )
        return elements

    def plot_trees(
        self,
        original_clf: DecisionTreeClassifier,
        simplified_clf: DecisionTreeClassifier,
        feature_names: list[str],
        class_names: list[str],
        save_path: str | None = None,
    ) -> str:
        """Create side-by-side visualization of original and simplified trees.

        Args:
            original_clf: Original decision tree classifier.
            simplified_clf: Simplified decision tree classifier.
            feature_names: Names of features.
            class_names: Names of classes.
            save_path: Optional path to save the figure. If None, uses default.

        Returns:
            Path where the figure was saved.
        """
        if save_path is None:
            save_path = self._get_save_path(Constants.FILE_DECISION_TREE_COMPARISON)

        fig, axes = plt.subplots(
            Constants.ONE,
            Constants.TWO,
            figsize=(Constants.FIGURE_WIDTH_20, Constants.FIGURE_WIDTH_10),
        )

        plot_tree(
            original_clf,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            ax=axes[Constants.ZERO],
            fontsize=Constants.FONTSIZE_SMALL,
        )
        axes[Constants.ZERO].set_title(
            f"Original Tree ({original_clf.tree_.node_count} nodes)",
            fontsize=Constants.FONTSIZE_TITLE,
            fontweight="bold",
        )

        plot_tree(
            simplified_clf,
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            ax=axes[Constants.ONE],
            fontsize=Constants.FONTSIZE_SMALL,
        )
        axes[Constants.ONE].set_title(
            f"Simplified Tree ({simplified_clf.tree_.node_count} nodes)",
            fontsize=Constants.FONTSIZE_TITLE,
            fontweight="bold",
        )

        plt.suptitle(
            Constants.TITLE_DECISION_TREE_SIMPLIFICATION,
            fontsize=Constants.TITLE_FONTSIZE_16,
            fontweight="bold",
            y=Constants.ZERO_FLOAT + Constants.ONE - Constants.TENTH * Constants.TWO,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=Constants.DEFAULT_DPI, bbox_inches=Constants.DEFAULT_BBOX_INCHES)
        print(f"Tree comparison saved to: {save_path}")
        plt.close(fig)

        return save_path

    def plot_prediction_scatter(
        self,
        original_clf: DecisionTreeClassifier,
        simplified_clf: DecisionTreeClassifier,
        X_test: ndarray,
        y_test: ndarray,
        class_names: list[str],
        save_path: str | None = None,
    ) -> str:
        """Create scatter plot comparing original vs simplified tree predictions.

        Each point represents a test sample, with original tree prediction on x-axis
        and simplified tree prediction on y-axis. Points on the diagonal indicate
        matching predictions.

        Args:
            original_clf: Original decision tree classifier.
            simplified_clf: Simplified decision tree classifier.
            X_test: Test features.
            y_test: Test labels.
            class_names: Names of classes.
            save_path: Optional path to save the figure. If None, uses default.

        Returns:
            Path where the figure was saved.
        """
        if save_path is None:
            save_path = self._get_save_path(Constants.FILE_DECISION_TREE_PREDICTION_SCATTER)

        orig_pred = original_clf.predict(X_test)
        simp_pred = simplified_clf.predict(X_test)
        n_samples = len(y_test)

        fig, ax = plt.subplots(figsize=(Constants.FIGURE_WIDTH_10, Constants.FIGURE_HEIGHT_8))

        x_jitter = self._add_jitter(orig_pred.astype(float), Constants.SCATTER_JITTER)
        y_jitter = self._add_jitter(
            simp_pred.astype(float), Constants.SCATTER_JITTER, seed=Constants.RANDOM_SEED_JITTER
        )

        colors = []
        for sample_idx in range(n_samples):
            if orig_pred[sample_idx] == simp_pred[sample_idx]:
                if orig_pred[sample_idx] == y_test[sample_idx]:
                    colors.append(Constants.COLOR_AGREE_CORRECT)
                else:
                    colors.append(Constants.COLOR_AGREE_WRONG)
            elif orig_pred[sample_idx] == y_test[sample_idx]:
                colors.append(Constants.COLOR_ORIGINAL)
            elif simp_pred[sample_idx] == y_test[sample_idx]:
                colors.append(Constants.COLOR_SIMPLIFIED)
            else:
                colors.append(Constants.COLOR_BOTH_WRONG_DIFF)

        ax.scatter(
            x_jitter,
            y_jitter,
            c=colors,
            alpha=Constants.DEFAULT_ALPHA,
            s=Constants.DEFAULT_POINT_SIZE,
            edgecolors=Constants.EDGE_COLOR_BLACK,
            linewidth=Constants.LINEWIDTH_HALF,
        )

        min_val, max_val = Constants.MINUS_HALF, len(class_names) - Constants.HALF
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            Constants.LINE_DASHED,
            linewidth=Constants.LINE_WIDTH_2,
            label=Constants.LABEL_PERFECT_AGREEMENT,
        )

        for class_idx in range(len(class_names)):
            ax.axvline(
                class_idx,
                color=Constants.EDGE_COLOR_GRAY,
                linestyle="-",
                linewidth=Constants.LINEWIDTH_HALF,
                alpha=Constants.HALF,
            )
            ax.axhline(
                class_idx,
                color=Constants.EDGE_COLOR_GRAY,
                linestyle="-",
                linewidth=Constants.LINEWIDTH_HALF,
                alpha=Constants.HALF,
            )

        ax.set_xlabel(Constants.LABEL_ORIGINAL_TREE_PREDICTION, fontsize=Constants.FONTSIZE_LABEL)
        ax.set_ylabel(Constants.LABEL_SIMPLIFIED_TREE_PREDICTION, fontsize=Constants.FONTSIZE_LABEL)
        ax.set_title(
            "Prediction Comparison: Original vs Simplified Decision Tree\n"
            f"Agreement: {np_sum(orig_pred == simp_pred)}/{len(y_test)} "
            f"({mean(orig_pred == simp_pred):.1%})",
            fontsize=Constants.FONTSIZE_TITLE,
            fontweight="bold",
        )

        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        legend_elements = self._create_legend_elements(
            labels=[
                Constants.LABEL_ORIGINAL_CORRECT,
                Constants.LABEL_SIMPLIFIED_CORRECT,
                Constants.LABEL_BOTH_AGREE_CORRECT,
                Constants.LABEL_BOTH_AGREE_WRONG,
                Constants.LABEL_BOTH_WRONG_DIFF,
            ],
            colors=[
                Constants.COLOR_ORIGINAL,
                Constants.COLOR_SIMPLIFIED,
                Constants.COLOR_AGREE_CORRECT,
                Constants.COLOR_AGREE_WRONG,
                Constants.COLOR_BOTH_WRONG_DIFF,
            ],
        )
        ax.legend(handles=legend_elements, loc="lower right", fontsize=Constants.FONTSIZE_LEGEND)

        ax.grid(Constants.TRUE, alpha=Constants.GRID_ALPHA)
        ax.set_aspect(Constants.ASPECT_EQUAL)
        plt.tight_layout()
        plt.savefig(save_path, dpi=Constants.DEFAULT_DPI, bbox_inches=Constants.DEFAULT_BBOX_INCHES)
        print(f"Prediction scatter plot saved to: {save_path}")
        plt.close(fig)

        return save_path

    def plot_log_loss_scatter(
        self,
        original_clf: DecisionTreeClassifier,
        simplified_clf: DecisionTreeClassifier,
        X_test: ndarray,
        y_test: ndarray,
        class_names: list[str],
        save_path: str | None = None,
    ) -> LogLossResult:
        """Create scatter plot comparing log loss per sample.

        Log loss (cross-entropy loss) measures the uncertainty of predictions
        based on predicted probabilities. Lower log loss indicates more
        confident and accurate predictions.

        Args:
            original_clf: Original decision tree classifier.
            simplified_clf: Simplified decision tree classifier.
            X_test: Test features.
            y_test: Test labels.
            class_names: Names of classes (unused, kept for API consistency).
            save_path: Optional path to save the figure. If None, uses default.

        Returns:
            LogLossResult containing save path and log loss values.
        """
        if save_path is None:
            save_path = self._get_save_path(Constants.FILE_DECISION_TREE_LOG_LOSS_SCATTER)

        orig_proba = original_clf.predict_proba(X_test)
        simp_proba = simplified_clf.predict_proba(X_test)

        n_samples = len(y_test)
        orig_sample_loss = -np_log(orig_proba[arange(n_samples), y_test] + Constants.LOG_EPSILON)
        simp_sample_loss = -np_log(simp_proba[arange(n_samples), y_test] + Constants.LOG_EPSILON)

        orig_log_loss_val = mean(orig_sample_loss)
        simp_log_loss_val = mean(simp_sample_loss)

        fig, ax = plt.subplots(figsize=(Constants.FIGURE_WIDTH_10, Constants.FIGURE_HEIGHT_8))

        x_jitter = self._add_jitter(orig_sample_loss, Constants.LOG_LOSS_JITTER)
        y_jitter = self._add_jitter(
            simp_sample_loss, Constants.LOG_LOSS_JITTER, seed=Constants.RANDOM_SEED_JITTER
        )

        colors = []
        for sample_idx in range(n_samples):
            if (
                simp_sample_loss[sample_idx]
                < orig_sample_loss[sample_idx] - Constants.LOG_LOSS_COMPARE_EPSILON
            ):
                colors.append(Constants.COLOR_SIMPLIFIED)
            elif (
                orig_sample_loss[sample_idx]
                < simp_sample_loss[sample_idx] - Constants.LOG_LOSS_COMPARE_EPSILON
            ):
                colors.append(Constants.COLOR_ORIGINAL)
            else:
                colors.append(Constants.COLOR_EQUAL)

        ax.scatter(
            x_jitter,
            y_jitter,
            c=colors,
            alpha=Constants.DEFAULT_ALPHA,
            s=Constants.DEFAULT_POINT_SIZE,
            edgecolors=Constants.EDGE_COLOR_BLACK,
            linewidth=Constants.LINEWIDTH_HALF,
        )

        min_loss = (
            min(np_min(orig_sample_loss), np_min(simp_sample_loss)) - Constants.PLOT_RANGE_PAD
        )
        max_loss = (
            max(np_max(orig_sample_loss), np_max(simp_sample_loss)) + Constants.PLOT_RANGE_PAD
        )
        ax.plot(
            [min_loss, max_loss],
            [min_loss, max_loss],
            Constants.LINE_DASHED,
            linewidth=Constants.LINE_WIDTH_2,
            label=Constants.LABEL_EQUAL_LOSS,
        )
        ax.grid(Constants.TRUE, alpha=Constants.GRID_ALPHA, linestyle="-")

        ax.set_xlabel(Constants.LABEL_ORIGINAL_TREE_LOG_LOSS, fontsize=Constants.FONTSIZE_LABEL)
        ax.set_ylabel(Constants.LABEL_SIMPLIFIED_TREE_LOG_LOSS, fontsize=Constants.FONTSIZE_LABEL)
        ax.set_title(
            f"Log Loss Comparison: Original vs Simplified Decision Tree\n"
            f"Original: {orig_log_loss_val:.4f} | Simplified: {simp_log_loss_val:.4f} "
            f"(Δ = {simp_log_loss_val - orig_log_loss_val:+.4f})",
            fontsize=Constants.FONTSIZE_TITLE,
            fontweight="bold",
        )
        ax.set_xlim(min_loss, max_loss)
        ax.set_ylim(min_loss, max_loss)

        legend_elements = self._create_legend_elements(
            labels=[
                Constants.LABEL_ORIGINAL_BETTER,
                Constants.LABEL_SIMPLIFIED_BETTER,
                Constants.LABEL_EQUAL,
            ],
            colors=[Constants.COLOR_ORIGINAL, Constants.COLOR_SIMPLIFIED, Constants.COLOR_EQUAL],
        )
        ax.legend(handles=legend_elements, loc="lower right", fontsize=Constants.FONTSIZE_LEGEND)

        plt.tight_layout()
        plt.savefig(save_path, dpi=Constants.DEFAULT_DPI, bbox_inches=Constants.DEFAULT_BBOX_INCHES)
        print(f"Log loss scatter plot saved to: {save_path}")
        plt.close(fig)

        return LogLossResult(
            save_path,
            orig_log_loss_val,
            simp_log_loss_val,
        )

    def plot_log_loss_curve(
        self,
        original_clf: DecisionTreeClassifier,
        simplified_clf: DecisionTreeClassifier,
        X_test: ndarray,
        y_test: ndarray,
        class_names: list[str],
        save_path: str | None = None,
    ) -> str:
        """Create log loss curve visualization with actual predictions overlaid.

        Shows the theoretical log loss curve (-log(p) for true class) with
        actual predictions from both models overlaid as scatter points.
        This visualization reveals how confident wrong predictions are
        heavily penalized.

        Args:
            original_clf: Original decision tree classifier.
            simplified_clf: Simplified decision tree classifier.
            X_test: Test features.
            y_test: Test labels.
            class_names: Names of classes.
            save_path: Optional path to save the figure. If None, uses default.

        Returns:
            Path where the figure was saved.
        """
        if save_path is None:
            save_path = self._get_save_path(Constants.FILE_DECISION_TREE_LOG_LOSS_CURVE)

        orig_proba = original_clf.predict_proba(X_test)
        simp_proba = simplified_clf.predict_proba(X_test)
        n_classes = len(class_names)

        fig, axes = plt.subplots(
            Constants.ONE,
            n_classes,
            figsize=(Constants.FIGURE_SCALE_5 * n_classes, Constants.FIGURE_SCALE_4),
        )
        if n_classes == Constants.ONE:
            axes = [axes]

        prob = linspace(Constants.PROBABILITY_MIN, Constants.PROBABILITY_MAX, Constants.HUNDRED)
        loss_curve = -np_log(prob)

        for class_idx, ax in enumerate(axes):
            ax.plot(
                prob,
                loss_curve,
                Constants.LINE_SOLID_GREEN,
                linewidth=Constants.LINE_WIDTH_2,
                alpha=0.7,
                label=Constants.LABEL_LOSS_CURVE,
            )
            ax.set_xlim(Constants.ZERO, Constants.ONE)
            ax.set_ylim(Constants.ZERO, max(loss_curve) + Constants.HALF)

            orig_class_proba = orig_proba[:, class_idx]
            simp_class_proba = simp_proba[:, class_idx]

            true_class_mask = y_test == class_idx
            other_class_mask = y_test != class_idx

            if np_any(true_class_mask):
                orig_loss_true = -np_log(orig_class_proba[true_class_mask] + Constants.LOG_EPSILON)
                ax.scatter(
                    orig_class_proba[true_class_mask],
                    orig_loss_true,
                    c=Constants.COLOR_ORIGINAL,
                    s=Constants.DEFAULT_SMALL_POINT_SIZE,
                    alpha=Constants.DEFAULT_ALPHA,
                    label="Original (true class)",
                    edgecolors=Constants.EDGE_COLOR_BLACK,
                    linewidth=Constants.DEFAULT_MARKER_EDGE_WIDTH,
                )
            if np_any(other_class_mask):
                orig_loss_other = -np_log(
                    Constants.ONE - orig_class_proba[other_class_mask] + Constants.LOG_EPSILON
                )
                ax.scatter(
                    orig_class_proba[other_class_mask],
                    orig_loss_other,
                    c=Constants.COLOR_ORIGINAL,
                    s=Constants.DEFAULT_SMALL_POINT_SIZE,
                    alpha=Constants.ALPHA_0_3,
                    marker=Constants.MARKER_X,
                    label="Original (other class)",
                )

            if np_any(true_class_mask):
                simp_loss_true = -np_log(simp_class_proba[true_class_mask] + Constants.LOG_EPSILON)
                ax.scatter(
                    simp_class_proba[true_class_mask] + Constants.PLOT_OFFSET,
                    simp_loss_true,
                    c=Constants.COLOR_SIMPLIFIED,
                    s=Constants.DEFAULT_SMALL_POINT_SIZE,
                    alpha=Constants.DEFAULT_ALPHA,
                    marker=Constants.MARKER_SQUARE,
                    label="Simplified (true class)",
                    edgecolors=Constants.EDGE_COLOR_BLACK,
                    linewidth=Constants.DEFAULT_MARKER_EDGE_WIDTH,
                )
            if np_any(other_class_mask):
                simp_loss_other = -np_log(
                    Constants.ONE - simp_class_proba[other_class_mask] + Constants.LOG_EPSILON
                )
                ax.scatter(
                    simp_class_proba[other_class_mask] + Constants.PLOT_OFFSET,
                    simp_loss_other,
                    c=Constants.COLOR_SIMPLIFIED,
                    s=Constants.DEFAULT_SMALL_POINT_SIZE,
                    alpha=Constants.ALPHA_0_3,
                    marker=Constants.MARKER_X,
                    label="Simplified (other class)",
                )

            ax.set_xlabel(Constants.LABEL_PREDICTED_PROBABILITY, fontsize=Constants.FONTSIZE_TICK)
            ax.set_ylabel(Constants.LABEL_LOG_LOSS_AXIS, fontsize=Constants.FONTSIZE_TICK)
            ax.set_title(
                f"Class: {class_names[class_idx]}",
                fontsize=Constants.FONTSIZE_SMALL,
                fontweight="bold",
            )
            ax.grid(Constants.TRUE, alpha=Constants.GRID_ALPHA)
            ax.legend(fontsize=Constants.LEGEND_FONTSIZE_8, loc="upper right")

        labels_list = list(range(n_classes))
        plt.suptitle(
            f"{Constants.TITLE_LOG_LOSS_CURVE_TREE}\n"
            f"Original: {log_loss(y_test, orig_proba, labels=labels_list):.4f} | "
            f"Simplified: {log_loss(y_test, simp_proba, labels=labels_list):.4f}",
            fontsize=Constants.FONTSIZE_TITLE,
            fontweight="bold",
            y=Constants.ONE_FLOAT + Constants.TENTH * Constants.TWO,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=Constants.DEFAULT_DPI, bbox_inches=Constants.DEFAULT_BBOX_INCHES)
        print(f"Log loss curve saved to: {save_path}")
        plt.close(fig)

        return save_path

    def plot_confusion_matrices(
        self,
        original_clf: DecisionTreeClassifier,
        simplified_clf: DecisionTreeClassifier,
        X_test: ndarray,
        y_test: ndarray,
        class_names: list[str],
        save_path: str | None = None,
    ) -> str:
        """Create side-by-side confusion matrices for original and simplified trees.

        Args:
            original_clf: Original decision tree classifier.
            simplified_clf: Simplified decision tree classifier.
            X_test: Test features.
            y_test: Test labels.
            class_names: Names of classes.
            save_path: Optional path to save the figure. If None, uses default.

        Returns:
            Path where the figure was saved.
        """
        if save_path is None:
            save_path = self._get_save_path(Constants.FILE_DECISION_TREE_CONFUSION_MATRIX)

        orig_pred = original_clf.predict(X_test)
        simp_pred = simplified_clf.predict(X_test)

        orig_cm = confusion_matrix(y_test, orig_pred, labels=list(range(len(class_names))))
        simp_cm = confusion_matrix(y_test, simp_pred, labels=list(range(len(class_names))))

        fig, axes = plt.subplots(
            Constants.ONE,
            Constants.TWO,
            figsize=(Constants.FIGURE_WIDTH_14, Constants.FIGURE_HEIGHT_6),
        )

        disp1 = ConfusionMatrixDisplay(confusion_matrix=orig_cm, display_labels=class_names)
        disp1.plot(ax=axes[Constants.ZERO], cmap=plt.cm.Blues, colorbar=Constants.TRUE)  # type: ignore[attr-defined]
        axes[Constants.ZERO].set_title(
            f"Original Tree\nAccuracy: {mean(orig_pred == y_test):.2%}",
            fontsize=Constants.FONTSIZE_TITLE,
            fontweight="bold",
        )

        disp2 = ConfusionMatrixDisplay(confusion_matrix=simp_cm, display_labels=class_names)
        disp2.plot(ax=axes[Constants.ONE], cmap=plt.cm.Blues, colorbar=Constants.TRUE)  # type: ignore[attr-defined]
        axes[Constants.ONE].set_title(
            f"Simplified Tree\nAccuracy: {mean(simp_pred == y_test):.2%}",
            fontsize=Constants.FONTSIZE_TITLE,
            fontweight="bold",
        )

        plt.suptitle(
            Constants.TITLE_CONFUSION_MATRICES_TREE,
            fontsize=Constants.TITLE_FONTSIZE_16,
            fontweight="bold",
            y=Constants.ZERO_FLOAT + Constants.ONE - Constants.TENTH * Constants.TWO,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=Constants.DEFAULT_DPI, bbox_inches=Constants.DEFAULT_BBOX_INCHES)
        print(f"Confusion matrix saved to: {save_path}")
        plt.close(fig)

        return save_path

    def plot_roc_curves(
        self,
        original_clf: DecisionTreeClassifier,
        simplified_clf: DecisionTreeClassifier,
        X_test: ndarray,
        y_test: ndarray,
        class_names: list[str],
        save_path: str | None = None,
    ) -> ROCResult:
        """Plot ROC curves for original and simplified trees.

        For multi-class classification, uses One-vs-Rest binarization.

        Args:
            original_clf: Original decision tree classifier.
            simplified_clf: Simplified decision tree classifier.
            X_test: Test features.
            y_test: Test labels.
            class_names: Names of classes.
            save_path: Optional path to save the figure. If None, uses default.

        Returns:
            ROCResult containing save path and AUC scores for both models.
        """
        if save_path is None:
            save_path = self._get_save_path(Constants.FILE_DECISION_TREE_ROC_CURVES)

        n_classes = len(class_names)

        orig_proba = original_clf.predict_proba(X_test)
        simp_proba = simplified_clf.predict_proba(X_test)

        y_test_binarized = zeros((len(y_test), n_classes))
        y_test_binarized[arange(len(y_test)), y_test] = Constants.ONE

        orig_auc: dict[str, float] = {}
        simp_auc: dict[str, float] = {}
        fpr_orig: dict[str, Any] = {}
        tpr_orig: dict[str, Any] = {}
        fpr_simp: dict[str, Any] = {}
        tpr_simp: dict[str, Any] = {}

        for class_idx, class_name in enumerate(class_names):
            y_col = y_test_binarized[:, class_idx]
            if np_any(y_col == Constants.ONE) and np_any(y_col == Constants.ZERO):
                fpr_orig[class_name], tpr_orig[class_name], _ = roc_curve(
                    y_col, orig_proba[:, class_idx]
                )
                fpr_simp[class_name], tpr_simp[class_name], _ = roc_curve(
                    y_col, simp_proba[:, class_idx]
                )
                orig_auc[class_name] = float(roc_auc_score(y_col, orig_proba[:, class_idx]))
                simp_auc[class_name] = float(roc_auc_score(y_col, simp_proba[:, class_idx]))
            else:
                fpr_orig[class_name] = zeros(Constants.TWO)
                tpr_orig[class_name] = zeros(Constants.TWO)
                fpr_simp[class_name] = zeros(Constants.TWO)
                tpr_simp[class_name] = zeros(Constants.TWO)
                orig_auc[class_name] = Constants.HALF
                simp_auc[class_name] = Constants.HALF

        fig, axes = plt.subplots(
            Constants.ONE,
            n_classes,
            figsize=(Constants.FIGURE_SCALE_5 * n_classes, Constants.FIGURE_SCALE_4),
        )
        if n_classes == Constants.ONE:
            axes = [axes]

        for idx, class_name in enumerate(class_names):
            ax = axes[idx]
            class_color = Constants.CLASS_COLORS[idx % len(Constants.CLASS_COLORS)]

            ax.plot(
                fpr_orig[class_name],
                tpr_orig[class_name],
                color=class_color,
                linewidth=Constants.DEFAULT_THICK_LINE_WIDTH,
                label=f"Original (AUC = {orig_auc[class_name]:.4f})",
            )

            ax.plot(
                fpr_simp[class_name],
                tpr_simp[class_name],
                color=class_color,
                linewidth=Constants.DEFAULT_THICK_LINE_WIDTH,
                linestyle="--",
                label=f"Simplified (AUC = {simp_auc[class_name]:.4f})",
            )

            ax.plot(
                [Constants.ZERO, Constants.ONE],
                [Constants.ZERO, Constants.ONE],
                Constants.LINE_DOTTED,
                linewidth=Constants.LINE_WIDTH_1,
                label="Random",
            )
            ax.set_xlabel(Constants.LABEL_FALSE_POSITIVE_RATE, fontsize=Constants.FONTSIZE_TICK)
            ax.set_ylabel(Constants.LABEL_TRUE_POSITIVE_RATE, fontsize=Constants.FONTSIZE_TICK)
            ax.set_title(
                f"Class: {class_name}", fontsize=Constants.FONTSIZE_SMALL, fontweight="bold"
            )
            ax.legend(loc="lower right", fontsize=Constants.FONTSIZE_LEGEND)
            ax.set_xlim([Constants.ZERO, Constants.ONE])
            ax.set_ylim([Constants.ZERO, Constants.ONE_POINT_ZERO_FIVE])
            ax.grid(alpha=Constants.GRID_ALPHA)

        plt.suptitle(
            f"{Constants.TITLE_ROC_CURVES_TREE}\n"
            f"Original Macro AUC: {mean(list(orig_auc.values())):.4f} | "
            f"Simplified Macro AUC: {mean(list(simp_auc.values())):.4f}",
            fontsize=Constants.TITLE_FONTSIZE_16,
            fontweight="bold",
            y=Constants.ZERO_FLOAT + Constants.ONE - Constants.TENTH * Constants.TWO,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=Constants.DEFAULT_DPI, bbox_inches=Constants.DEFAULT_BBOX_INCHES)
        print(f"ROC curve saved to: {save_path}")
        plt.close(fig)

        return ROCResult(
            save_path,
            {"original": orig_auc, "simplified": simp_auc},
        )
