"""Tests for the TreeExtractor class."""

from sklearn.datasets import load_diabetes
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from tests.constants import ONE
from tests.constants import ONE_FLOAT
from tests.constants import TWO
from tests.constants import ZERO_FLOAT
from tests.constants import TestConstants
from tests.constants import TestEdgeCaseConstants
from tests.constants import TestFeatureConstants
from tree_graph import TreeExtractor
from tree_graph.constants import Constants
from tree_graph.constants import NumberConstants
from tree_graph.rule_set import RuleSet
from tree_graph.tree_converter import TreeConverter


class TestTreeExtractor:
    """Tests for the TreeExtractor class."""

    def test_parse_leaf_value_with_invalid_string(self) -> None:
        """Test _parse_leaf_value with malformed string."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_THREE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        TreeConverter(RuleSet.create(Constants.RULE_SET_BASIC))._create_estimator_from_data(
            {
                "children_left": [NumberConstants.MINUS_ONE],
                "children_right": [NumberConstants.MINUS_ONE],
                "feature": [NumberConstants.MINUS_TWO],
                "threshold": [ZERO_FLOAT],
                "value": [[ZERO_FLOAT]],
                "impurity": [ZERO_FLOAT],
                "n_node_samples": [ONE],
                "weighted_n_node_samples": [ONE_FLOAT],
            },
            clf,
        )
        extractor_tree = TreeExtractor(clf)

        # Test with invalid value string
        result = extractor_tree._parse_leaf_value(TestEdgeCaseConstants.INVALID_LEAF_STRING)
        assert result == ZERO_FLOAT

        # Test with empty string
        result = extractor_tree._parse_leaf_value(TestEdgeCaseConstants.EMPTY_LEAF_STRING)
        assert result == ZERO_FLOAT

    def test_parse_if_with_invalid_threshold(self) -> None:
        """Test _parse_if when threshold cannot be parsed."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_THREE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        extractor = TreeExtractor(clf)

        # Manually test the parsing with invalid threshold
        term_str = TestEdgeCaseConstants.INVALID_IF_STRING

        parsed = extractor._parse_if(term_str)
        assert parsed.threshold == ZERO_FLOAT  # Should default to 0.0

    def test_find_feature_index_not_found(self) -> None:
        """Test _find_feature_index when feature is not found."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_THREE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        extractor = TreeExtractor(clf)

        # Test with unknown feature name
        result = extractor._find_feature_index(TestFeatureConstants.UNKNOWN_FEATURE_NAME)
        assert result == NumberConstants.ZERO  # Should default to 0

    def test_find_feature_index_with_feature_prefix(self) -> None:
        """Test _find_feature_index with feature_N format."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_THREE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        extractor = TreeExtractor(clf)

        # Test with feature_N format
        result = extractor._find_feature_index(TestFeatureConstants.FEATURE_TWO_FULL_NAME)
        assert result == TWO

        # Test with invalid feature_N format
        result = extractor._find_feature_index(TestFeatureConstants.INVALID_FEATURE_NAME)
        assert result == NumberConstants.ZERO  # Should default to 0

    def test_compute_subtree_impurity_regression(self) -> None:
        """Test _compute_subtree_impurity for regression (no classes_)."""
        features, labels = load_diabetes(return_X_y=True)
        reg = DecisionTreeRegressor(
            max_depth=TestConstants.MAX_DEPTH_THREE,
            random_state=TestConstants.RANDOM_SEED,
        )
        reg.fit(features, labels)

        extractor = TreeExtractor(reg)

        # For regression, should return 0.0
        result = extractor._compute_subtree_impurity(NumberConstants.ZERO)
        assert result == ZERO_FLOAT

    def test_compute_subtree_impurity_with_zero_total(self) -> None:
        """Test _compute_subtree_impurity when class counts sum to zero."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_THREE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        extractor = TreeExtractor(clf)

        # Manually set up zero samples
        extractor.children_left = [NumberConstants.MINUS_ONE]
        extractor.children_right = [NumberConstants.MINUS_ONE]
        extractor.features = [NumberConstants.MINUS_TWO]
        extractor.thresholds = [ZERO_FLOAT]
        extractor.values = [[ZERO_FLOAT]]
        extractor.impurities = [ZERO_FLOAT]
        extractor.n_node_samples = [NumberConstants.ZERO]  # Zero samples
        extractor.weighted_n_node_samples = [ZERO_FLOAT]

        result = extractor._compute_subtree_impurity(NumberConstants.ZERO)
        assert result == ZERO_FLOAT

    def test_fallback_leaf_creation(self) -> None:
        """Test that fallback creates leaf for unrecognized term format."""
        features, labels = load_iris(return_X_y=True)
        clf = DecisionTreeClassifier(
            max_depth=TestConstants.MAX_DEPTH_THREE,
            random_state=TestConstants.RANDOM_SEED,
        )
        clf.fit(features, labels)

        extractor = TreeExtractor(clf)

        # Test with unrecognized term format
        result = extractor._extract_node_str(TestEdgeCaseConstants.UNRECOGNIZED_TERM_STRING)
        assert result == NumberConstants.ZERO  # Should create fallback leaf at index 0
