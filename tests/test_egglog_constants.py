"""Tests for EgglogConstants class."""

from tree_graph.constants import EgglogConstants


class TestEgglogConstants:
    """Tests for the EgglogConstants class."""

    def test_variable_names(self) -> None:
        """Test egglog variable name constants are defined."""
        assert EgglogConstants.VAR_CONDITION == "condition"
        assert EgglogConstants.VAR_LEAF == "leaf"
        assert EgglogConstants.VAR_FEATURE == "feature"

    def test_branch_variables(self) -> None:
        """Test branch variable constants are defined."""
        assert EgglogConstants.VAR_THEN_BRANCH == "then_branch"
        assert EgglogConstants.VAR_ELSE_BRANCH == "else_branch"
