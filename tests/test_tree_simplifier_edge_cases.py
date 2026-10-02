"""Tests for edge cases in tree simplification."""

from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from tests.constants import TestConstants
from tests.constants import TestFeatureConstants
from tree_graph import TreeSimplifier


class TestTreeSimplifierEdgeCases:
    """Tests for edge cases in the TreeSimplifier class."""

    def test_simplify_preserves_feature_names(self) -> None:
        """Test that simplification preserves feature_names_in_."""

        # Create data with named features
        features, labels = load_iris(return_X_y=True)
        feature_names = [
            TestFeatureConstants.IRIS_FEATURE_SEPAL_LENGTH,
            TestFeatureConstants.IRIS_FEATURE_SEPAL_WIDTH,
            TestFeatureConstants.IRIS_FEATURE_PETAL_LENGTH,
            TestFeatureConstants.IRIS_FEATURE_PETAL_WIDTH,
        ]
        features_df = DataFrame(features, columns=feature_names)

        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_THREE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features_df, labels)

        # Verify feature names are set
        assert hasattr(clf, "feature_names_in_")

        simplifier = TreeSimplifier()
        simplified = simplifier.simplify(clf, features_df, labels)

        # Should preserve feature names
        assert hasattr(simplified, "feature_names_in_")
        assert list(simplified.feature_names_in_) == feature_names
