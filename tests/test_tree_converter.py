"""Tests for tree conversion."""

from numpy import array_equal
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from tests.constants import MAX_ITERATIONS_TEN
from tests.constants import TestConstants
from tree_graph import TreeConverter
from tree_graph.constants import NumberConstants
from tree_graph.rule_set import RuleSet
from tree_graph.tree import Tree


class TestTreeConverter:
    """Tests for the TreeConverter class."""

    def test_to_egraph_classifier(self) -> None:
        """Test converting a classifier to egraph."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_TWO,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        rule_set = RuleSet.basic()
        converter = TreeConverter(rule_set)
        result = converter.to_egraph(clf)

        assert result.egraph is not None
        assert result.root_term is not None
        assert isinstance(result.root_term, Tree)

    def test_to_egraph_preserves_structure(self) -> None:
        """Test that conversion preserves tree structure."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_TWO,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        rule_set = RuleSet.basic()
        converter = TreeConverter(rule_set)
        result = converter.to_egraph(clf)

        # The root term should be an If node for a non-trivial tree
        term_str = str(result.root_term)
        assert term_str.startswith("Tree.if_")

    def test_to_egraph_leaf_only_tree(self) -> None:
        """Test converting a single-leaf tree (stump)."""
        features, labels = load_iris(return_X_y=True)
        # max_depth=1 is the minimum allowed, creates a simple split
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_ONE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        rule_set = RuleSet.basic()
        converter = TreeConverter(rule_set)
        result = converter.to_egraph(clf)

        # Should be an If node with two leaf children
        term_str = str(result.root_term)
        assert term_str.startswith("Tree.if_")

    def test_from_egraph_classifier(self) -> None:
        """Test extracting a classifier from egraph."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_TWO,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        rule_set = RuleSet.basic()
        converter = TreeConverter(rule_set)
        result = converter.to_egraph(clf)
        result.egraph.run(MAX_ITERATIONS_TEN)

        extracted_clf = converter.from_egraph(result.egraph, clf, result.root_term)

        assert isinstance(extracted_clf, DecisionTreeClassifier)
        assert extracted_clf.tree_.node_count > NumberConstants.ZERO

    def test_from_egraph_preserves_predictions(self) -> None:
        """Test that extraction preserves predictions."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_TWO,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        rule_set = RuleSet.basic()
        converter = TreeConverter(rule_set)
        result = converter.to_egraph(clf)
        result.egraph.run(MAX_ITERATIONS_TEN)

        extracted_clf = converter.from_egraph(result.egraph, clf, result.root_term)

        orig_pred = clf.predict(features)
        extracted_pred = extracted_clf.predict(features)

        assert array_equal(orig_pred, extracted_pred)

    def test_from_egraph_metadata(self) -> None:
        """Test that extraction preserves metadata."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_TWO,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        rule_set = RuleSet.basic()
        converter = TreeConverter(rule_set)
        result = converter.to_egraph(clf)
        result.egraph.run(MAX_ITERATIONS_TEN)

        extracted_clf = converter.from_egraph(result.egraph, clf, result.root_term)

        assert array_equal(extracted_clf.classes_, clf.classes_)
        assert extracted_clf.n_features_in_ == clf.n_features_in_
        assert extracted_clf.n_outputs_ == clf.n_outputs_

    def test_roundtrip_classifier(self) -> None:
        """Test roundtrip conversion for classifier."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_TWO,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        # Convert to egraph and back
        rule_set = RuleSet.basic()
        converter = TreeConverter(rule_set)
        result = converter.to_egraph(clf)
        extracted_clf = converter.from_egraph(result.egraph, clf, result.root_term)

        # Predictions must match
        orig_pred = clf.predict(features)
        extracted_pred = extracted_clf.predict(features)

        assert array_equal(orig_pred, extracted_pred)

    def test_roundtrip_preserves_node_count(self) -> None:
        """Test that roundtrip preserves node count (without simplification)."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_TWO,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        original_nodes = clf.tree_.node_count

        # Convert to egraph and back without running rules
        rule_set = RuleSet.basic()
        converter = TreeConverter(rule_set)
        result = converter.to_egraph(clf)
        # Don't run saturation - just extract immediately
        extracted_clf = converter.from_egraph(result.egraph, clf, result.root_term)

        # Node count should be the same
        assert extracted_clf.tree_.node_count == original_nodes
