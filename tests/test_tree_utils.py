"""Tests for the TreeUtils class."""

from numpy import array
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from tests.constants import ONE_FLOAT
from tests.constants import ZERO_FLOAT
from tests.constants import TestConstants
from tests.constants import TestEdgeCaseConstants
from tests.constants import TestFeatureConstants
from tree_graph.tree_utils import TreeUtils


class TestTreeUtils:
    """Tests for the TreeUtils class."""

    def test_get_feature_names_with_feature_names_in(self) -> None:
        """Test TreeUtils.get_feature_names when feature_names_in_ exists."""

        features, labels = load_iris(return_X_y=True)
        feature_names_list = [
            TestFeatureConstants.IRIS_FEATURE_SEPAL_LENGTH,
            TestFeatureConstants.IRIS_FEATURE_SEPAL_WIDTH,
            TestFeatureConstants.IRIS_FEATURE_PETAL_LENGTH,
            TestFeatureConstants.IRIS_FEATURE_PETAL_WIDTH,
        ]
        features_df = DataFrame(features, columns=feature_names_list)

        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_THREE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features_df, labels)

        # Verify feature_names_in_ exists
        assert hasattr(clf, "feature_names_in_")

        utils = TreeUtils()
        feature_names = utils.get_feature_names(clf)

        # Should have mapped all feature indices to names
        assert len(feature_names) == TestConstants.N_FEATURES_FOUR
        assert TestFeatureConstants.FEATURE_INDEX_ZERO in feature_names
        assert (
            feature_names[TestFeatureConstants.FEATURE_INDEX_ZERO]
            == TestFeatureConstants.IRIS_FEATURE_SEPAL_LENGTH
        )

    def test_compute_gini_with_zero_total(self) -> None:
        """Test TreeUtils.compute_gini when total is zero."""
        utils = TreeUtils()
        value_row = array([ZERO_FLOAT, ZERO_FLOAT, ZERO_FLOAT])
        gini = utils.compute_gini(value_row)
        assert gini == ZERO_FLOAT

    def test_compute_gini_with_valid_values(self) -> None:
        """Test TreeUtils.compute_gini with valid class counts."""
        utils = TreeUtils()
        pure_value = array([TestEdgeCaseConstants.GINI_PURE_VALUE_1, ZERO_FLOAT, ZERO_FLOAT])
        gini = utils.compute_gini(pure_value)
        assert gini == ZERO_FLOAT

        mixed_value = array(
            [
                TestEdgeCaseConstants.GINI_MIXED_VALUE_1,
                TestEdgeCaseConstants.GINI_MIXED_VALUE_1,
                ZERO_FLOAT,
            ]
        )
        gini = utils.compute_gini(mixed_value)
        assert gini > ZERO_FLOAT
        assert gini < ONE_FLOAT
