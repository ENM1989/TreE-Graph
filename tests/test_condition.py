"""Tests for the Condition class."""

from egglog import String
from egglog import f64

from tests.constants import TestFeatureConstants
from tests.constants import TestNodeConstants
from tree_graph.condition import Condition


class TestCondition:
    """Tests for the Condition class."""

    def test_creation(self) -> None:
        """Test creating a Condition."""
        cond = Condition(
            String(TestFeatureConstants.FEATURE_ZERO_FULL_NAME),
            f64(TestNodeConstants.THRESHOLD_ZERO_POINT_FIVE),
        )
        assert cond is not None

    def test_string_representation(self) -> None:
        """Test Condition string representation."""
        cond = Condition(
            String(TestFeatureConstants.FEATURE_ZERO_FULL_NAME),
            f64(TestNodeConstants.THRESHOLD_ZERO_POINT_FIVE),
        )
        term_str = str(cond)
        assert "Condition" in term_str
