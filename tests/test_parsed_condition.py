"""Tests for ParsedCondition dataclass."""

from tests.constants import TWO
from tests.constants import ZERO_FLOAT
from tree_graph import ParsedCondition
from tree_graph.constants import NumberConstants


class TestParsedCondition:
    """Tests for the ParsedCondition class."""

    def test_creation(self) -> None:
        """Test creating a ParsedCondition."""
        parsed = ParsedCondition(
            feature_idx=NumberConstants.ZERO,
            threshold=NumberConstants.HALF,
            then_term="Tree.leaf(0)",
            else_term="Tree.leaf(1)",
        )

        assert parsed.feature_idx == NumberConstants.ZERO
        assert parsed.threshold == NumberConstants.HALF
        assert parsed.then_term == "Tree.leaf(0)"
        assert parsed.else_term == "Tree.leaf(1)"

    def test_dataclass_fields(self) -> None:
        """Test that all fields are accessible."""
        parsed = ParsedCondition(
            feature_idx=TWO,
            threshold=1.5,
            then_term="Tree.if_(...)",
            else_term="Tree.leaf(2)",
        )

        # All fields should be accessible
        assert hasattr(parsed, "feature_idx")
        assert hasattr(parsed, "threshold")
        assert hasattr(parsed, "then_term")
        assert hasattr(parsed, "else_term")

    def test_with_negative_feature_index(self) -> None:
        """Test ParsedCondition with negative feature index (edge case)."""
        parsed = ParsedCondition(
            feature_idx=NumberConstants.MINUS_ONE,
            threshold=ZERO_FLOAT,
            then_term="Tree.leaf(0)",
            else_term="Tree.leaf(0)",
        )

        assert parsed.feature_idx == NumberConstants.MINUS_ONE
        assert parsed.threshold == ZERO_FLOAT
