"""Internal utility classes for tree conversion."""

from typing import Any

from numpy import ndarray
from numpy import sum as np_sum
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from .constants import Constants
from .constants import NumberConstants


class TreeUtils:
    """Utility class for tree conversion operations.

    This class provides static-like methods for common tree conversion
    operations such as extracting feature names and computing impurity.
    """

    def get_feature_names(
        self,
        estimator: DecisionTreeClassifier | DecisionTreeRegressor,
    ) -> dict[int, str]:
        """Extract feature names from estimator.

        Args:
            estimator: Trained decision tree estimator.

        Returns:
            Dictionary mapping feature indices to names.
        """
        try:
            feature_names = estimator.feature_names_in_
            return {idx: str(name) for idx, name in enumerate(feature_names)}
        except AttributeError:
            return {}

    def is_leaf_node(self, tree_data: dict[str, Any], node_idx: int) -> bool:
        """Check if a node is a leaf.

        Args:
            tree_data: Tree data dictionary with children_left.
            node_idx: Index of the node to check.

        Returns:
            True if the node is a leaf (left child is -1).
        """
        return bool(tree_data["children_left"][node_idx] == NumberConstants.MINUS_ONE)

    def compute_gini(self, class_counts: ndarray) -> float:
        """Compute Gini impurity from class counts.

        Args:
            class_counts: Array of shape (n_classes,) containing class counts.

        Returns:
            Gini impurity value between 0 and 1.
        """
        total = float(np_sum(class_counts))
        if total == 0:
            return NumberConstants.ZERO_FLOAT
        proportions = class_counts / total
        return float(NumberConstants.ONE_FLOAT - np_sum(proportions**NumberConstants.TWO))

    @classmethod
    def find_child_terms(cls, tree_str: str) -> list[str]:
        """Find child Tree terms in a string.

        Parses a string containing egglog Tree terms and extracts
        balanced parenthesized sub-terms.

        Args:
            tree_str: String containing Tree terms.

        Returns:
            List of Tree term strings.
        """
        terms: list[str] = []
        depth = NumberConstants.ZERO
        start = NumberConstants.MINUS_ONE
        in_term = False

        pos = NumberConstants.ZERO
        while pos < len(tree_str):
            # Look for "Tree." to start a new term
            prefix_len = Constants.EXTRACTOR_PREFIX_LENGTH
            if tree_str[pos : pos + prefix_len] == Constants.EGRAPH_TREE_PREFIX and not in_term:
                start = pos
                in_term = True
                depth = NumberConstants.ZERO
                pos += prefix_len
                continue

            if in_term:
                if tree_str[pos] == "(":
                    depth += NumberConstants.ONE
                elif tree_str[pos] == ")":
                    depth -= NumberConstants.ONE
                    if depth == NumberConstants.ZERO and start >= NumberConstants.ZERO:
                        terms.append(tree_str[start : pos + NumberConstants.ONE])
                        in_term = False
                        start = NumberConstants.MINUS_ONE
            pos += NumberConstants.ONE

        return terms
