"""Tests for the RuleSet class."""

from pytest import raises

from tests.constants import ONE
from tests.constants import RULE_SET_INVALID
from tree_graph.constants import Constants
from tree_graph.rule_set import RuleSet


class TestRuleSet:
    """Tests for the RuleSet class."""

    _FULL_RULE_SET_COUNT: int = 5  # redundant: 1, dead_branch: 2, subtree: 2

    def test_create_basic(self) -> None:
        """Test creating a basic rule set."""
        rules = RuleSet.create(Constants.RULE_SET_BASIC)
        assert rules.name == Constants.RULE_SET_BASIC
        assert len(rules) == ONE

    def test_create_full(self) -> None:
        """Test creating a full rule set."""
        rules = RuleSet.create(Constants.RULE_SET_FULL)
        assert rules.name == Constants.RULE_SET_FULL
        assert len(rules) == self._FULL_RULE_SET_COUNT

    def test_create_invalid_name(self) -> None:
        """Test that creating a rule set with invalid name raises ValueError."""
        with raises(ValueError, match="Unknown rule set"):
            RuleSet.create(RULE_SET_INVALID)  # type: ignore[arg-type]

    def test_basic_method(self) -> None:
        """Test the basic class method."""
        rules = RuleSet.basic()
        assert rules.name == Constants.RULE_SET_BASIC
        assert len(rules) == ONE

    def test_full_method(self) -> None:
        """Test the full class method."""
        rules = RuleSet.full()
        assert rules.name == Constants.RULE_SET_FULL
        assert len(rules) == self._FULL_RULE_SET_COUNT

    def test_len(self) -> None:
        """Test the __len__ method."""
        rules = RuleSet()
        assert len(rules) == 0
        rules.add_redundant_split_elimination()
        assert len(rules) == ONE

    def test_iter(self) -> None:
        """Test the __iter__ method."""
        rules = RuleSet()
        rules.add_redundant_split_elimination()
        rule_list = list(rules)
        assert len(rule_list) == ONE

    def test_repr(self) -> None:
        """Test the __repr__ method."""
        rules = RuleSet(name="test")
        rules.add_redundant_split_elimination()
        repr_str = repr(rules)
        assert "test" in repr_str
        assert "rules=1" in repr_str

    def test_add_redundant_split_elimination(self) -> None:
        """Test adding the redundant split elimination rule."""
        rules = RuleSet()
        rules.add_redundant_split_elimination()
        assert len(rules) == ONE

    def test_method_chaining(self) -> None:
        """Test that add_redundant_split_elimination returns self for chaining."""
        rules = RuleSet()
        result = rules.add_redundant_split_elimination()
        assert result is rules

    def test_add_dead_branch_pruning(self) -> None:
        """Test adding the dead branch pruning rule."""
        rules = RuleSet()
        rules.add_dead_branch_pruning()
        assert len(rules) == 2  # Two rules are added

    def test_full_rule_set_includes_dead_branch_pruning(self) -> None:
        """Test that full rule set includes dead branch pruning."""
        rules = RuleSet.create(Constants.RULE_SET_FULL)
        assert len(rules) >= 2  # At least dead branch pruning rules

    def test_add_subtree_replacement(self) -> None:
        """Test adding the subtree replacement rule."""
        rules = RuleSet()
        rules.add_subtree_replacement()
        assert len(rules) == 2  # Two rules are added

    def test_full_rule_set_includes_subtree_replacement(self) -> None:
        """Test that full rule set includes subtree replacement."""
        rules = RuleSet.create(Constants.RULE_SET_FULL)
        assert len(rules) >= 4  # All rules combined

    def test_full_has_more_rules_than_basic(self) -> None:
        """Test that full rule set has more rules than basic."""
        basic = RuleSet.create(Constants.RULE_SET_BASIC)
        full = RuleSet.create(Constants.RULE_SET_FULL)
        assert len(full) > len(basic)

    def test_full_rule_count(self) -> None:
        """Test the total number of rules in the full set."""
        full = RuleSet.create(Constants.RULE_SET_FULL)
        assert len(full) == self._FULL_RULE_SET_COUNT
