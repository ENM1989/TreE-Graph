"""Tests for the TreeSimplifier class."""

from numpy import array_equal
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from tests.constants import MAX_ITERATIONS_FIFTY
from tests.constants import MAX_ITERATIONS_HUNDRED
from tests.constants import MAX_ITERATIONS_ONE
from tests.constants import MAX_ITERATIONS_TEN
from tests.constants import TestConstants
from tree_graph import TreeSimplifier
from tree_graph.constants import NumberConstants


class TestTreeSimplifier:
    """Tests for the TreeSimplifier class."""

    def test_simplify_classifier_preserves_predictions(self) -> None:
        """Test that simplification preserves predictions on training data."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_FIVE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        simplifier = TreeSimplifier()
        simplified_clf = simplifier.simplify(clf, features, labels)

        orig_pred = clf.predict(features)
        simp_pred = simplified_clf.predict(features)

        assert array_equal(orig_pred, simp_pred)

    def test_simplify_classifier_preserves_test_predictions(self) -> None:
        """Test that simplification preserves predictions on test data."""
        features, labels = load_iris(return_X_y=True)
        features_train, features_test, labels_train, labels_test = train_test_split(
            features,
            labels,
            test_size=TestConstants.TEST_SIZE_THIRTY_PERCENT,
            random_state=TestConstants.RANDOM_SEED,
        )

        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_FIVE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features_train, labels_train)

        simplifier = TreeSimplifier()
        simplified_clf = simplifier.simplify(clf, features_train, labels_train)

        orig_pred = clf.predict(features_test)
        simp_pred = simplified_clf.predict(features_test)

        assert array_equal(orig_pred, simp_pred)

    def test_simplify_classifier_preserves_accuracy(self) -> None:
        """Test that simplification preserves accuracy."""
        features, labels = load_iris(return_X_y=True)
        features_train, features_test, labels_train, labels_test = train_test_split(
            features,
            labels,
            test_size=TestConstants.TEST_SIZE_THIRTY_PERCENT,
            random_state=TestConstants.RANDOM_SEED,
        )

        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_FIVE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features_train, labels_train)

        simplifier = TreeSimplifier()
        simplified_clf = simplifier.simplify(clf, features_train, labels_train)

        orig_accuracy = clf.score(features_test, labels_test)
        simp_accuracy = simplified_clf.score(features_test, labels_test)

        assert orig_accuracy == simp_accuracy

    def test_simplify_reduces_nodes(self) -> None:
        """Test that simplification can reduce tree size."""
        # Create a dataset where simplification should help
        features, labels = load_iris(return_X_y=True)

        # Train a deeper tree that likely has redundant splits
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_TEN,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        simplifier = TreeSimplifier()
        simplified_clf = simplifier.simplify(clf, features, labels)

        # The simplified tree should have <= nodes than original
        assert simplified_clf.tree_.node_count <= clf.tree_.node_count

    def test_simplify_regressor_preserves_predictions(self) -> None:
        """Test that simplification preserves regression predictions."""
        features, labels = load_diabetes(return_X_y=True)
        reg = DecisionTreeRegressor(
            max_depth=TestConstants.MAX_DEPTH_FIVE,
            random_state=TestConstants.RANDOM_SEED,
        )
        reg.fit(features, labels)

        simplifier = TreeSimplifier()
        simplified_reg = simplifier.simplify(reg, features, labels)

        orig_pred = reg.predict(features)
        simp_pred = simplified_reg.predict(features)

        assert array_equal(orig_pred, simp_pred)

    def test_simplify_regressor_preserves_test_predictions(self) -> None:
        """Test that simplification preserves regression predictions on test data."""
        features, labels = load_diabetes(return_X_y=True)
        features_train, features_test, labels_train, labels_test = train_test_split(
            features,
            labels,
            test_size=TestConstants.TEST_SIZE_THIRTY_PERCENT,
            random_state=TestConstants.RANDOM_SEED,
        )

        reg = DecisionTreeRegressor(
            max_depth=TestConstants.MAX_DEPTH_FIVE,
            random_state=TestConstants.RANDOM_SEED,
        )
        reg.fit(features_train, labels_train)

        simplifier = TreeSimplifier()
        simplified_reg = simplifier.simplify(reg, features_train, labels_train)

        orig_pred = reg.predict(features_test)
        simp_pred = simplified_reg.predict(features_test)

        assert array_equal(orig_pred, simp_pred)

    def test_simplify_single_node_tree(self) -> None:
        """Test simplification of a simple tree (stump)."""
        features, labels = load_iris(return_X_y=True)

        # Train a simple tree (max_depth=1 is minimum allowed)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_ONE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        simplifier = TreeSimplifier()
        simplified_clf = simplifier.simplify(clf, features, labels)

        # Should still work
        orig_pred = clf.predict(features)
        simp_pred = simplified_clf.predict(features)

        assert array_equal(orig_pred, simp_pred)

    def test_simplify_max_iterations(self) -> None:
        """Test that max_iterations parameter works."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_FIVE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        # Test with different iteration counts
        for max_iter in [
            MAX_ITERATIONS_ONE,
            MAX_ITERATIONS_TEN,
            MAX_ITERATIONS_HUNDRED,
        ]:
            simplifier = TreeSimplifier(max_iterations=max_iter)
            simplified_clf = simplifier.simplify(clf, features, labels)
            orig_pred = clf.predict(features)
            simp_pred = simplified_clf.predict(features)
            assert array_equal(orig_pred, simp_pred)

    def test_simplify_with_stats_returns_correct_stats(self) -> None:
        """Test that simplify_with_stats returns correct statistics."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_FIVE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        simplifier = TreeSimplifier(max_iterations=MAX_ITERATIONS_FIFTY)
        result = simplifier.simplify_with_stats(clf, features, labels)

        assert result.original_nodes == clf.tree_.node_count
        assert result.simplified_nodes == result.estimator.tree_.node_count
        assert result.iterations_run == MAX_ITERATIONS_FIFTY
        assert result.predictions_preserved is True
        assert result.reduction_percent >= NumberConstants.ZERO_FLOAT

    def test_simplify_with_stats_preserves_predictions(self) -> None:
        """Test that simplify_with_stats preserves predictions."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_FIVE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        simplifier = TreeSimplifier()
        result = simplifier.simplify_with_stats(clf, features, labels)

        orig_pred = clf.predict(features)
        simp_pred = result.estimator.predict(features)

        assert array_equal(orig_pred, simp_pred)
        assert result.predictions_preserved is True
