"""Main entry point for tree simplification."""

from typing import Literal
from typing import cast
from warnings import warn

from numpy import array_equal
from numpy import ndarray
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from .constants import Constants
from .constants import NumberConstants
from .rule_set import RuleSet
from .simplification_result import SimplificationResult
from .tree_converter import TreeConverter


class TreeSimplifier:
    """Simplifies scikit-learn decision trees using e-graph rewriting.

    This class converts a trained scikit-learn decision tree to an e-graph
    representation, applies rewrite rules to discover equivalent tree structures,
    and extracts the minimal equivalent tree.

    Attributes:
        max_iterations: Maximum e-graph saturation iterations.
        rule_set: The rule set to use for simplification.
        validation: Whether to validate predictions after simplification.
        strict: If True, raise ValueError on prediction mismatch instead of returning original.

    Example:
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.datasets import load_iris
        >>> from tree_graph import TreeSimplifier
        >>> X, y = load_iris(return_X_y=True)
        >>> clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        >>> clf.fit(X, y)
        >>> simplifier = TreeSimplifier(max_iterations=100)
        >>> simplified_clf = simplifier.simplify(clf, X, y)
        >>> print(f"Original: {clf.tree_.node_count} nodes")
        >>> print(f"Simplified: {simplified_clf.tree_.node_count} nodes")
    """

    def __init__(
        self,
        max_iterations: int = Constants.DEFAULT_MAX_ITERATIONS,
        rule_set: Literal["basic", "full"] = cast(
            Literal["basic", "full"], Constants.RULE_SET_BASIC
        ),
        validation: bool = True,
        strict: bool = False,
    ):
        """Initialize the TreeSimplifier.

        Args:
            max_iterations: Maximum e-graph saturation iterations. Default is 100.
            rule_set: The rule set to use ("basic" or "full"). Default is "basic".
            validation: Whether to validate predictions after simplification. Default is True.
            strict: Whether to raise ValueError on prediction mismatch. Default is False.
        """
        self.max_iterations = max_iterations
        self.rule_set = RuleSet.create(rule_set)
        self.validation = validation
        self.strict = strict

    def simplify(
        self,
        estimator: DecisionTreeClassifier | DecisionTreeRegressor,
        X: ndarray,
        labels: ndarray,
    ) -> DecisionTreeClassifier | DecisionTreeRegressor:
        """Simplify a trained decision tree using e-graph rewriting.

        Args:
            estimator: Trained DecisionTreeClassifier or DecisionTreeRegressor.
            X: Training features (used for validation).
            labels: Training labels (used for validation).

        Returns:
            Simplified decision tree estimator that produces equivalent predictions.

        Raises:
            ValueError: If strict is True and predictions don't match.
        """
        # Convert tree to e-graph
        converter = TreeConverter(self.rule_set)
        result = converter.to_egraph(estimator)
        egraph = result.egraph
        root_term = result.root_term

        # Run saturation to apply rewrite rules
        egraph.run(self.max_iterations)

        # Extract the simplified tree
        simplified_estimator = converter.from_egraph(egraph, estimator, root_term)

        # Validate that predictions are preserved
        if self.validation:
            orig_predictions = estimator.predict(X)
            simp_predictions = simplified_estimator.predict(X)

            if not array_equal(orig_predictions, simp_predictions):
                if self.strict:
                    raise ValueError(Constants.SIMPLIFICATION_PREDICTION_MISMATCH_MESSAGE)
                warn(
                    Constants.SIMPLIFICATION_PREDICTION_MISMATCH_MESSAGE,
                    UserWarning,
                    stacklevel=2,
                )
                return estimator

        return simplified_estimator

    def simplify_with_stats(
        self,
        estimator: DecisionTreeClassifier | DecisionTreeRegressor,
        X: ndarray,
        labels: ndarray,
    ) -> SimplificationResult:
        """Simplify a tree and return statistics about the simplification.

        Args:
            estimator: Trained DecisionTreeClassifier or DecisionTreeRegressor.
            X: Training features (used for validation).
            labels: Training labels (used for validation).

        Returns:
            SimplificationResult containing the simplified estimator and
            simplification statistics.
        """
        original_nodes = estimator.tree_.node_count

        simplified_estimator = self.simplify(estimator, X, labels)
        simplified_nodes = simplified_estimator.tree_.node_count

        reduction_percent = (
            NumberConstants.ONE - simplified_nodes / original_nodes
        ) * NumberConstants.PERCENTAGE

        orig_predictions = estimator.predict(X)
        simp_predictions = simplified_estimator.predict(X)
        predictions_preserved = array_equal(orig_predictions, simp_predictions)

        return SimplificationResult(
            simplified_estimator,
            original_nodes,
            simplified_nodes,
            reduction_percent,
            self.max_iterations,
            predictions_preserved,
        )


def simplify_tree(
    estimator: DecisionTreeClassifier | DecisionTreeRegressor,
    X: ndarray,
    y: ndarray,
    max_iterations: int = Constants.DEFAULT_MAX_ITERATIONS,
    rule_set: Literal["basic", "full"] = "basic",
    validation: bool = True,
    strict: bool = False,
) -> DecisionTreeClassifier | DecisionTreeRegressor:
    """Simplify a trained scikit-learn decision tree using e-graph rewriting.

    Convenience function that wraps TreeSimplifier.

    Args:
        estimator: Trained DecisionTreeClassifier or DecisionTreeRegressor.
        X: Training features (used for validation).
        y: Training labels (used for validation).
        max_iterations: Maximum e-graph saturation iterations. Default is 100.
        rule_set: The rule set to use ("basic" or "full"). Default is "basic".
        validation: Whether to validate predictions after simplification. Default is True.
        strict: Whether to raise ValueError on prediction mismatch. Default is False.

    Returns:
        Simplified decision tree estimator with equivalent predictions.
    """
    simplifier = TreeSimplifier(
        max_iterations=max_iterations,
        rule_set=rule_set,
        validation=validation,
        strict=strict,
    )
    return simplifier.simplify(estimator, X, y)
