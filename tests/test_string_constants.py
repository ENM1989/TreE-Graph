"""Tests for StringConstants class."""

from tree_graph.constants import StringConstants


class TestStringConstants:
    """Tests for the StringConstants class."""

    def test_tree_structure_strings(self) -> None:
        """Test tree structure string constants are defined."""
        assert StringConstants.LEFT == "left"
        assert StringConstants.RIGHT == "right"

    def test_zero_string(self) -> None:
        """Test zero string representation is defined."""
        assert StringConstants.ZERO_STRING == "0.0"

    def test_default_feature(self) -> None:
        """Test default feature string is defined."""
        assert StringConstants.DEFAULT_FEATURE == "f0"
