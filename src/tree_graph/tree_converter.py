"""Convert scikit-learn decision trees to egglog e-graphs."""

from pickle import dumps
from pickle import loads
from typing import Any

from egglog import EGraph
from egglog import String
from egglog import f64
from numpy import argmax
from numpy import array
from numpy import float64
from numpy import isclose
from numpy import max as np_max
from numpy import ndarray
from numpy import zeros
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree._tree import Tree as SklearnTree

from .condition import Condition
from .constants import NumberConstants
from .constants import StringConstants
from .conversion_result import ConversionResult
from .rule_set import RuleSet
from .tree import Tree
from .tree_utils import TreeUtils


class TreeConverter:
    """Converts between scikit-learn decision trees and egglog e-graphs.

    This class handles the conversion of trained scikit-learn decision trees
    to egglog e-graph representation and back, preserving tree structure and
    predictions.

    Attributes:
        rule_set: The rule set to register with the e-graph.

    Example:
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from tree_graph import TreeConverter, RuleSet
        >>> clf = DecisionTreeClassifier(max_depth=3)
        >>> clf.fit(X, y)
        >>> converter = TreeConverter(RuleSet.create("basic"))
        >>> result = converter.to_egraph(clf)
        >>> result.egraph.run(100)
        >>> simplified = converter.from_egraph(result.egraph, clf, result.root_term)
    """

    def __init__(self, rule_set: RuleSet):
        """Initialize the TreeConverter.

        Args:
            rule_set: The rule set to register with the e-graph.
        """
        self.rule_set = rule_set

    def to_egraph(
        self, estimator: DecisionTreeClassifier | DecisionTreeRegressor
    ) -> ConversionResult:
        """Convert a scikit-learn decision tree to an egglog e-graph.

        Args:
            estimator: A trained DecisionTreeClassifier or DecisionTreeRegressor.

        Returns:
            ConversionResult containing the EGraph and root Tree term.
        """
        egraph = EGraph()

        # Register simplification rules
        for rule in self.rule_set.rules:
            egraph.register(rule)

        tree = estimator.tree_
        utils = TreeUtils()
        feature_names = utils.get_feature_names(estimator)

        # Convert the tree starting from the root
        root_term = self._convert_node(tree, NumberConstants.ZERO, feature_names)

        # Add the root term to the egraph
        egraph.register(root_term)

        return ConversionResult(egraph, root_term)

    def from_egraph(
        self,
        egraph: EGraph,
        original_estimator: DecisionTreeClassifier | DecisionTreeRegressor,
        root_term: Tree,
    ) -> DecisionTreeClassifier | DecisionTreeRegressor:
        """Extract a simplified decision tree from an e-graph.

        Args:
            egraph: A saturated e-graph containing tree terms.
            original_estimator: The original estimator to use as a template.
            root_term: The root Tree term to extract.

        Returns:
            A new decision tree estimator with the simplified structure.
        """
        from .tree_extractor import TreeExtractor

        # Extract the optimal term from the egraph
        extracted = egraph.extract(root_term)

        # Convert the extracted term back to a tree structure
        extractor = TreeExtractor(original_estimator)
        tree_data = extractor.extract(extracted)

        # Create a new estimator with the simplified tree
        return self._create_estimator_from_data(tree_data, original_estimator)

    def _convert_node(self, tree: SklearnTree, node_id: int, feature_names: dict[int, str]) -> Tree:
        """Convert a single tree node to an egglog term.

        Args:
            tree: The sklearn tree object.
            node_id: The node index in the tree.
            feature_names: Mapping from feature index to name.

        Returns:
            An egglog Tree term representing the node.
        """
        if tree.children_left[node_id] == tree.children_right[node_id]:
            value = tree.value[node_id].flatten()
            if len(value) == NumberConstants.ONE:
                float_value = float(value[NumberConstants.ZERO])
            else:
                float_value = float(argmax(value))
            return Tree.leaf(f64(float_value))

        feature_idx = tree.feature[node_id]
        threshold = float(tree.threshold[node_id])
        feature_name = feature_names.get(feature_idx, f"feature_{feature_idx}")

        condition = Condition(String(feature_name), f64(threshold))

        left_child = self._convert_node(tree, tree.children_left[node_id], feature_names)
        right_child = self._convert_node(tree, tree.children_right[node_id], feature_names)

        return Tree.if_(condition, left_child, right_child)

    def _create_estimator_from_data(
        self,
        tree_data: dict[str, Any],
        original_estimator: DecisionTreeClassifier | DecisionTreeRegressor,
    ) -> DecisionTreeClassifier | DecisionTreeRegressor:
        """Create a new estimator from tree data using sklearn's Tree state API.

        Args:
            tree_data: Dictionary containing tree structure data.
            original_estimator: Original estimator to copy metadata from.

        Returns:
            New decision tree estimator with the simplified structure.
        """
        orig_tree = original_estimator.tree_
        orig_state = orig_tree.__getstate__()

        n_nodes = len(tree_data["feature"])

        if n_nodes == NumberConstants.ZERO:
            n_nodes = NumberConstants.ONE
            tree_data["children_left"] = [NumberConstants.MINUS_ONE]
            tree_data["children_right"] = [NumberConstants.MINUS_ONE]
            tree_data["feature"] = [NumberConstants.MINUS_ONE - NumberConstants.ONE]
            tree_data["threshold"] = [NumberConstants.ZERO_FLOAT]
            tree_data["value"] = [[NumberConstants.ZERO_FLOAT]]
            tree_data["impurity"] = [NumberConstants.ZERO_FLOAT]
            tree_data["n_node_samples"] = [NumberConstants.ONE]
            tree_data["weighted_n_node_samples"] = [NumberConstants.ONE_FLOAT]

        nodes_dtype = orig_state["nodes"].dtype
        nodes = zeros(n_nodes, dtype=nodes_dtype)

        for node_idx in range(n_nodes):
            nodes[node_idx] = (
                tree_data["children_left"][node_idx],
                tree_data["children_right"][node_idx],
                tree_data["feature"][node_idx],
                tree_data["threshold"][node_idx],
                tree_data["impurity"][node_idx],
                tree_data["n_node_samples"][node_idx],
                tree_data["weighted_n_node_samples"][node_idx],
                NumberConstants.ZERO,
            )

        n_outputs = orig_tree.n_outputs
        is_regressor = not hasattr(original_estimator, "classes_")

        if is_regressor:
            value_array = zeros((n_nodes, n_outputs, NumberConstants.ONE), dtype=float64)
            for node_idx, value in enumerate(tree_data["value"]):
                if len(value) > NumberConstants.ZERO:
                    value_array[node_idx, NumberConstants.ZERO, NumberConstants.ZERO] = value[
                        NumberConstants.ZERO
                    ]
        else:
            n_classes_per_output = int(orig_tree.n_classes[NumberConstants.ZERO])
            value_array = zeros((n_nodes, n_outputs, n_classes_per_output), dtype=float64)

            utils = TreeUtils()
            for node_idx in range(n_nodes):
                if utils.is_leaf_node(tree_data, node_idx):
                    predicted_class = int(tree_data["value"][node_idx][NumberConstants.ZERO])
                    if NumberConstants.ZERO <= predicted_class < n_classes_per_output:
                        n_samples = tree_data["n_node_samples"][node_idx] or NumberConstants.ONE
                        value_array[node_idx, NumberConstants.ZERO, predicted_class] = float(
                            n_samples
                        )

            for node_idx in range(
                n_nodes - NumberConstants.ONE,
                NumberConstants.MINUS_ONE,
                NumberConstants.MINUS_ONE,
            ):
                if utils.is_leaf_node(tree_data, node_idx):
                    continue
                left, right = (
                    tree_data["children_left"][node_idx],
                    tree_data["children_right"][node_idx],
                )
                if left >= NumberConstants.ZERO and right >= NumberConstants.ZERO:
                    value_array[node_idx, NumberConstants.ZERO, :] = (
                        value_array[left, NumberConstants.ZERO, :]
                        + value_array[right, NumberConstants.ZERO, :]
                    )

        self._map_nodes_and_copy_stats(
            orig_tree, tree_data, nodes, value_array, is_regressor=is_regressor
        )

        impurities = zeros(n_nodes)
        utils = TreeUtils()
        for node_idx in range(n_nodes):
            if is_regressor:
                impurities[node_idx] = NumberConstants.ZERO_FLOAT
            else:
                impurities[node_idx] = utils.compute_gini(
                    value_array[node_idx, NumberConstants.ZERO, :]
                )

        # Update impurity field in each node
        for node_idx in range(n_nodes):
            nodes[node_idx]["impurity"] = impurities[node_idx]

        # Create state dictionary
        state = {
            "max_depth": int(
                np_max(array(tree_data["children_left"]) != NumberConstants.MINUS_ONE)
            ),
            "node_count": n_nodes,
            "nodes": nodes,
            "values": value_array,
        }

        # Preserve n_classes format that sklearn expects
        tree_bytes = dumps(orig_tree)
        new_tree = loads(tree_bytes)
        new_tree.__setstate__(state)

        # Create new estimator of the same type
        new_estimator = type(original_estimator)()
        new_estimator.tree_ = new_tree

        # Copy metadata
        if hasattr(original_estimator, "classes_"):
            new_estimator.classes_ = original_estimator.classes_.copy()
        if hasattr(original_estimator, "n_features_in_"):
            new_estimator.n_features_in_ = original_estimator.n_features_in_
        if hasattr(original_estimator, "feature_names_in_"):
            new_estimator.feature_names_in_ = original_estimator.feature_names_in_.copy()
        if hasattr(original_estimator, "n_outputs_"):
            new_estimator.n_outputs_ = original_estimator.n_outputs_
        if hasattr(original_estimator, "n_classes_"):
            new_estimator.n_classes_ = original_estimator.n_classes_

        return new_estimator

    def _map_nodes_and_copy_stats(
        self,
        orig_tree: SklearnTree,
        tree_data: dict[str, Any],
        nodes: ndarray,
        value_array: ndarray,
        is_regressor: bool = False,
    ) -> None:
        """Map simplified tree nodes to original tree and copy sample statistics.

        Args:
            orig_tree: Original sklearn tree.
            tree_data: Simplified tree data.
            nodes: Nodes array to update (modified in place).
            value_array: Value array to update (modified in place).
            is_regressor: Whether this is a regression model.
        """
        n_nodes = len(tree_data["feature"])
        n_classes = (
            value_array.shape[NumberConstants.TWO]
            if len(value_array.shape) > NumberConstants.TWO
            else NumberConstants.ONE
        )

        for node_idx in range(n_nodes):
            is_leaf = tree_data["children_left"][node_idx] == NumberConstants.MINUS_ONE

            orig_idx = self._find_matching_original_node(orig_tree, tree_data, node_idx)

            if orig_idx >= NumberConstants.ZERO:
                nodes[node_idx]["n_node_samples"] = orig_tree.n_node_samples[orig_idx]
                nodes[node_idx]["weighted_n_node_samples"] = orig_tree.weighted_n_node_samples[
                    orig_idx
                ]

            if is_leaf:
                if is_regressor:
                    # In regression, value_array already contains exact continuous prediction
                    continue

                predicted_class = int(tree_data["value"][node_idx][NumberConstants.ZERO])
                if orig_idx >= NumberConstants.ZERO:
                    orig_val = orig_tree.value[orig_idx, NumberConstants.ZERO, :]
                    if argmax(orig_val) == predicted_class:
                        value_array[node_idx, NumberConstants.ZERO, :] = orig_val
                    else:
                        value_array[node_idx, NumberConstants.ZERO, :] = NumberConstants.ZERO_FLOAT
                        value_array[node_idx, NumberConstants.ZERO, predicted_class] = float(
                            nodes[node_idx]["n_node_samples"] or NumberConstants.ONE
                        )
                else:
                    if NumberConstants.ZERO <= predicted_class < n_classes:
                        value_array[node_idx, NumberConstants.ZERO, :] = NumberConstants.ZERO_FLOAT
                        value_array[node_idx, NumberConstants.ZERO, predicted_class] = (
                            NumberConstants.ONE_FLOAT
                        )

        if not is_regressor:
            # Recompute internal node values from weighted children average
            for node_idx in range(
                n_nodes - NumberConstants.ONE, NumberConstants.MINUS_ONE, NumberConstants.MINUS_ONE
            ):
                if tree_data["children_left"][node_idx] == NumberConstants.MINUS_ONE:
                    continue  # Skip leaves
                left = tree_data["children_left"][node_idx]
                right = tree_data["children_right"][node_idx]
                if left >= NumberConstants.ZERO and right >= NumberConstants.ZERO:
                    left_n = nodes[left]["n_node_samples"]
                    right_n = nodes[right]["n_node_samples"]
                    total = left_n + right_n
                    if total > NumberConstants.ZERO:
                        value_array[node_idx, NumberConstants.ZERO, :] = (
                            left_n * value_array[left, NumberConstants.ZERO, :]
                            + right_n * value_array[right, NumberConstants.ZERO, :]
                        ) / total

    def _build_node_mapping(
        self, orig_tree: SklearnTree, tree_data: dict[str, Any]
    ) -> dict[tuple[int, float], int]:
        """Build a mapping from original tree nodes to simplified tree nodes.

        Args:
            orig_tree: Original sklearn tree.
            tree_data: Simplified tree data.

        Returns:
            Dictionary mapping (feature, threshold) tuples to simplified node indices.
        """
        mapping: dict[tuple[int, float], int] = {}
        n_nodes = len(tree_data["feature"])

        for node_idx in range(n_nodes):
            feature = tree_data["feature"][node_idx]
            threshold = tree_data["threshold"][node_idx]
            if feature >= NumberConstants.ZERO:  # Internal node
                mapping[(feature, threshold)] = node_idx

        return mapping

    def _find_matching_original_node(
        self,
        orig_tree: SklearnTree,
        tree_data: dict[str, Any],
        simp_node_idx: int,
        orig_mapping: dict[tuple[int, float], int] | None = None,
    ) -> int:
        """Find the matching node in the original tree for a simplified tree node.

        Traces the decision path from root downwards in both trees simultaneously.

        Args:
            orig_tree: Original sklearn tree.
            tree_data: Simplified tree data.
            simp_node_idx: Index of the node in simplified tree.
            orig_mapping: Optional mapping for compatibility.

        Returns:
            Index of matching node in original tree, or -1 if not found.
        """
        if simp_node_idx < NumberConstants.ZERO or simp_node_idx >= len(tree_data["feature"]):
            return NumberConstants.MINUS_ONE

        # Trace path from root to simp_node_idx in simplified tree
        path: list[tuple[int, float, str]] = []
        current = NumberConstants.ZERO
        while current != simp_node_idx:
            if current >= len(tree_data["feature"]):
                return NumberConstants.MINUS_ONE
            feature = tree_data["feature"][current]
            threshold = tree_data["threshold"][current]
            if feature < NumberConstants.ZERO:
                return NumberConstants.MINUS_ONE

            left_child = tree_data["children_left"][current]
            right_child = tree_data["children_right"][current]

            if left_child == simp_node_idx or (
                left_child >= NumberConstants.ZERO
                and self._is_ancestor_of(left_child, simp_node_idx, tree_data)
            ):
                path.append((feature, threshold, StringConstants.LEFT))
                current = left_child
            elif right_child == simp_node_idx or (
                right_child >= NumberConstants.ZERO
                and self._is_ancestor_of(right_child, simp_node_idx, tree_data)
            ):
                path.append((feature, threshold, StringConstants.RIGHT))
                current = right_child
            else:
                return NumberConstants.MINUS_ONE

        # Traverse original tree following matching splits
        orig_current = NumberConstants.ZERO
        for feature, threshold, direction in path:
            if orig_current < NumberConstants.ZERO or orig_current >= orig_tree.node_count:
                break
            if orig_tree.children_left[orig_current] == orig_tree.children_right[orig_current]:
                break

            orig_feat = orig_tree.feature[orig_current]
            orig_thresh = float(orig_tree.threshold[orig_current])

            if orig_feat == feature and isclose(orig_thresh, threshold):
                if direction == StringConstants.LEFT:
                    orig_current = orig_tree.children_left[orig_current]
                else:
                    orig_current = orig_tree.children_right[orig_current]
            else:
                # Path diverged (e.g. split was simplified/removed)
                break

        return orig_current

    def _is_ancestor_of(self, ancestor: int, descendant: int, tree_data: dict[str, Any]) -> bool:
        """Check if ancestor node is an ancestor of descendant node.

        Args:
            ancestor: Potential ancestor node index.
            descendant: Potential descendant node index.
            tree_data: Tree data structure.

        Returns:
            True if ancestor is an ancestor of descendant.
        """
        if ancestor < NumberConstants.ZERO or ancestor >= len(tree_data["feature"]):
            return False

        left = tree_data["children_left"][ancestor]
        right = tree_data["children_right"][ancestor]

        if left == descendant or right == descendant:
            return True

        if left >= NumberConstants.ZERO and self._is_ancestor_of(left, descendant, tree_data):
            return True
        if right >= NumberConstants.ZERO and self._is_ancestor_of(right, descendant, tree_data):
            return True

        return False
