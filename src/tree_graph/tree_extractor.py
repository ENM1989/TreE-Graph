"""Extract tree structure from egglog Tree terms."""

from typing import Any

from numpy import ndarray
from numpy import zeros
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from .constants import Constants
from .constants import NumberConstants
from .constants import StringConstants
from .parsed_condition import ParsedCondition
from .tree import Tree
from .tree_utils import TreeUtils


class TreeExtractor:
    """Helper class to extract tree structure from egglog Tree terms."""

    def __init__(self, original_estimator: DecisionTreeClassifier | DecisionTreeRegressor):
        """Initialize the TreeExtractor.

        Args:
            original_estimator: Original estimator for metadata and feature names.
        """
        self.original_estimator = original_estimator
        self._utils = TreeUtils()
        self.feature_names = self._utils.get_feature_names(original_estimator)
        self.original_tree = original_estimator.tree_

        # Lists to collect tree structure
        self.children_left: list[int] = []
        self.children_right: list[int] = []
        self.features: list[int] = []
        self.thresholds: list[float] = []
        self.values: list[list[float]] = []
        self.impurities: list[float] = []
        self.n_node_samples: list[int] = []
        self.weighted_n_node_samples: list[float] = []

    def extract(self, term: Tree) -> dict[str, Any]:
        """Extract tree data from an egglog Tree term.

        Args:
            term: The Tree term to extract.

        Returns:
            Dictionary containing tree data in sklearn format.
        """
        self._extract_node(term)

        return {
            "children_left": self.children_left,
            "children_right": self.children_right,
            "feature": self.features,
            "threshold": self.thresholds,
            "value": self.values,
            "impurity": self.impurities,
            "n_node_samples": self.n_node_samples,
            "weighted_n_node_samples": self.weighted_n_node_samples,
        }

    def _extract_node(self, term: Tree) -> int:
        """Extract a single node, return its index.

        Args:
            term: The Tree term to extract.

        Returns:
            Index of the extracted node.
        """
        term_str = str(term)
        return self._extract_node_str(term_str)

    def _extract_node_str(self, term_str: str) -> int:
        """Extract a single node from its string representation.

        Args:
            term_str: String representation of a Tree term.

        Returns:
            Index of the extracted node.
        """
        if term_str.startswith(f"{Constants.EGRAPH_TREE_PREFIX}leaf"):
            # Leaf node - extract value
            predicted_class = self._parse_leaf_value(term_str)
            return self._create_leaf(predicted_class)

        elif term_str.startswith(f"{Constants.EGRAPH_TREE_PREFIX}if_"):
            # Internal node - parse condition and children
            parsed = self._parse_if(term_str)

            # Reserve space for this node
            node_idx = len(self.children_left)
            self.children_left.append(NumberConstants.MINUS_ONE)  # Placeholder
            self.children_right.append(NumberConstants.MINUS_ONE)  # Placeholder
            self.features.append(parsed.feature_idx)
            self.thresholds.append(parsed.threshold)
            self.values.append([NumberConstants.ZERO_FLOAT])  # Placeholder for internal nodes
            self.impurities.append(NumberConstants.ZERO_FLOAT)  # Placeholder
            self.n_node_samples.append(NumberConstants.ZERO)  # Placeholder
            self.weighted_n_node_samples.append(NumberConstants.ZERO_FLOAT)  # Placeholder

            # Extract children recursively
            then_idx = self._extract_node_str(parsed.then_term)
            else_idx = self._extract_node_str(parsed.else_term)

            # Update placeholders
            self.children_left[node_idx] = then_idx
            self.children_right[node_idx] = else_idx

            # Compute impurity and sample counts from children
            self._compute_node_stats(node_idx, then_idx, else_idx)

            return node_idx

        # Fallback: create a leaf node
        return self._create_leaf(NumberConstants.ZERO_FLOAT)

    def _parse_leaf_value(self, term_str: str) -> float:
        """Parse the value from a leaf term string.

        Args:
            term_str: String representation of Tree.leaf(...)

        Returns:
            The leaf value as float.
        """
        # Extract value from "Tree.leaf(value)"
        try:
            leaf_prefix = f"{Constants.EGRAPH_TREE_PREFIX}leaf("
            value_str = term_str.replace(leaf_prefix, "").replace(")", "")
            return float(value_str)
        except (ValueError, AttributeError):
            return NumberConstants.ZERO_FLOAT

    def _parse_if(self, term_str: str) -> ParsedCondition:
        """Parse an If term string.

        Args:
            term_str: String representation of Tree.if_(...)

        Returns:
            ParsedCondition with parsed condition components.
        """
        # Format: Tree.if_(Condition("feature", threshold), then_branch, else_branch)

        # Find the Condition part
        cond_start = term_str.find("Condition(")
        cond_end = term_str.find(")", cond_start)
        cond_str = term_str[cond_start + len("Condition(") : cond_end]

        # Parse feature and threshold from Condition
        parts = cond_str.split(",", 1)
        feature_str = parts[0].strip().strip("\"'")
        threshold_str = (
            parts[1].strip() if len(parts) > NumberConstants.ONE else StringConstants.ZERO_STRING
        )

        # Find feature index
        feature_idx = self._find_feature_index(feature_str)

        try:
            threshold = float(threshold_str)
        except ValueError:
            threshold = NumberConstants.ZERO_FLOAT

        # Find the child terms after the condition
        remaining = term_str[cond_end + NumberConstants.ONE :]  # Skip past "Condition(...)"

        # Find then_branch and else_branch
        child_terms = TreeUtils.find_child_terms(remaining)

        then_term = (
            child_terms[NumberConstants.ZERO]
            if len(child_terms) > NumberConstants.ZERO
            else Constants.EGRAPH_LEAF_EXPRESSION
        )
        else_term = (
            child_terms[NumberConstants.ONE]
            if len(child_terms) > NumberConstants.ONE
            else Constants.EGRAPH_LEAF_EXPRESSION
        )

        return ParsedCondition(
            feature_idx,
            threshold,
            then_term,
            else_term,
        )

    def _find_feature_index(self, feature_str: str) -> int:
        """Find feature index from feature string.

        Args:
            feature_str: Feature name or identifier.

        Returns:
            Feature index.
        """
        # Try to find in feature names
        for idx, name in self.feature_names.items():
            if name == feature_str:
                return idx

        # Try to parse index from "feature_N" format
        if feature_str.startswith("feature_"):
            try:
                return int(feature_str.replace("feature_", ""))
            except ValueError:
                pass

        # Default to 0
        return NumberConstants.ZERO

    def _compute_node_stats(self, node_idx: int, then_idx: int, else_idx: int) -> None:
        """Compute impurity and sample counts for an internal node from its children.

        Args:
            node_idx: Index of the internal node.
            then_idx: Index of the then/left child.
            else_idx: Index of the else/right child.
        """
        # Sum sample counts from children
        n_samples = self.n_node_samples[then_idx] + self.n_node_samples[else_idx]
        weighted_samples = (
            self.weighted_n_node_samples[then_idx] + self.weighted_n_node_samples[else_idx]
        )

        self.n_node_samples[node_idx] = n_samples
        self.weighted_n_node_samples[node_idx] = weighted_samples

        # For impurity, we need to compute the class distribution at this node
        # by combining the class counts from both children
        # Since we only store predicted class at leaves, we approximate by
        # counting how many samples go to each predicted class in the subtree
        self.impurities[node_idx] = self._compute_subtree_impurity(node_idx)

    def _compute_subtree_impurity(self, node_idx: int) -> float:
        """Compute Gini impurity for a node based on class distribution in its subtree.

        Args:
            node_idx: Index of the node.

        Returns:
            Gini impurity value.
        """
        if not hasattr(self.original_estimator, "classes_"):
            return NumberConstants.ZERO_FLOAT

        n_classes = len(self.original_estimator.classes_)
        class_counts = zeros(n_classes)
        self._count_classes_in_subtree(node_idx, class_counts)

        return self._utils.compute_gini(class_counts)

    def _count_classes_in_subtree(self, node_idx: int, class_counts: ndarray) -> None:
        """Recursively count predicted classes in the subtree.

        Args:
            node_idx: Index of the current node.
            class_counts: Array to accumulate class counts.
        """
        is_leaf = self.children_left[node_idx] == NumberConstants.MINUS_ONE

        if is_leaf:
            # This is a leaf - add its samples to the predicted class
            predicted_class = int(self.values[node_idx][NumberConstants.ZERO])
            n_samples = self.n_node_samples[node_idx]
            if NumberConstants.ZERO <= predicted_class < len(class_counts):
                class_counts[predicted_class] += n_samples
        else:
            # Internal node - recurse to children
            left_child = self.children_left[node_idx]
            right_child = self.children_right[node_idx]
            if left_child >= NumberConstants.ZERO:
                self._count_classes_in_subtree(left_child, class_counts)
            if right_child >= NumberConstants.ZERO:
                self._count_classes_in_subtree(right_child, class_counts)

    def _create_leaf(self, predicted_class: float) -> int:
        """Create a leaf node.

        Args:
            predicted_class: The predicted class index at this leaf.

        Returns:
            Index of the created leaf node.
        """
        self.children_left.append(NumberConstants.MINUS_ONE)
        self.children_right.append(NumberConstants.MINUS_ONE)
        self.features.append(NumberConstants.MINUS_ONE - NumberConstants.ONE)  # -2 indicates leaf
        self.thresholds.append(NumberConstants.ZERO_FLOAT)
        self.values.append([predicted_class])

        # For leaves, compute impurity based on the predicted class
        # A pure leaf (all samples one class) has impurity 0.0
        # Since we don't have actual sample counts, assume purity for the predicted class
        self.impurities.append(NumberConstants.ZERO_FLOAT)  # Pure leaf
        self.n_node_samples.append(NumberConstants.ONE)  # Minimum sample count
        self.weighted_n_node_samples.append(NumberConstants.ONE_FLOAT)

        return len(self.children_left) - NumberConstants.ONE
