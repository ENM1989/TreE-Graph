"""Tests for SimplificationResult dataclass."""

from sklearn.tree import DecisionTreeClassifier

from tests.constants import TestDataclassConstants
from tree_graph import SimplificationResult


class TestSimplificationResult:
    """Tests for the SimplificationResult class."""

    def test_creation(self) -> None:
        """Test creating a SimplificationResult."""
        clf = DecisionTreeClassifier()
        result = SimplificationResult(
            estimator=clf,
            original_nodes=TestDataclassConstants.ORIGINAL_NODES_TEN,
            simplified_nodes=TestDataclassConstants.SIMPLIFIED_NODES_SEVEN,
            reduction_percent=TestDataclassConstants.REDUCTION_PERCENT_THIRTY,
            iterations_run=TestDataclassConstants.ITERATIONS_RUN_HUNDRED,
            predictions_preserved=True,
        )

        assert result.estimator is clf
        assert result.original_nodes == TestDataclassConstants.ORIGINAL_NODES_TEN
        assert result.simplified_nodes == TestDataclassConstants.SIMPLIFIED_NODES_SEVEN
        assert result.reduction_percent == TestDataclassConstants.REDUCTION_PERCENT_THIRTY
        assert result.iterations_run == TestDataclassConstants.ITERATIONS_RUN_HUNDRED
        assert result.predictions_preserved is True

    def test_dataclass_fields(self) -> None:
        """Test that all fields are accessible."""
        clf = DecisionTreeClassifier()
        result = SimplificationResult(
            estimator=clf,
            original_nodes=TestDataclassConstants.ORIGINAL_NODES_HUNDRED,
            simplified_nodes=TestDataclassConstants.SIMPLIFIED_NODES_EIGHTY,
            reduction_percent=TestDataclassConstants.REDUCTION_PERCENT_TWENTY,
            iterations_run=TestDataclassConstants.ITERATIONS_RUN_FIFTY,
            predictions_preserved=False,
        )

        # All fields should be accessible
        assert hasattr(result, "estimator")
        assert hasattr(result, "original_nodes")
        assert hasattr(result, "simplified_nodes")
        assert hasattr(result, "reduction_percent")
        assert hasattr(result, "iterations_run")
        assert hasattr(result, "predictions_preserved")
