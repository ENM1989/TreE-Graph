"""Egglog rewrite rules for decision tree simplification."""

from __future__ import annotations

from typing import Any
from typing import Literal

from egglog import *

from .condition import Condition
from .constants import Constants
from .constants import EgglogConstants
from .tree import Tree


class RuleSet:
    """A collection of egglog rewrite rules for tree simplification.

    This class manages a set of rewrite rules that can be applied to
    e-graphs containing decision tree terms.

    Attributes:
        rules: List of rewrite rules.
        name: Name of the rule set.

    Example:
        >>> rules = RuleSet.create("basic")
        >>> for rule in rules.rules:
        ...     egraph.register(rule)
    """

    def __init__(self, name: str = Constants.RULE_SET_BASIC):
        """Initialize an empty rule set.

        Args:
            name: Optional name for the rule set.
        """
        self.name = name
        self._rules: list[Any] = []

    @property
    def rules(self) -> list[Any]:
        """Get the list of rewrite rules.

        Returns:
            List of rewrite rules.
        """
        return self._rules

    def add_redundant_split_elimination(self) -> RuleSet:
        """Add the redundant split elimination rule.

        This rule collapses splits where both branches lead to identical outcomes:
        If(condition, tree, tree) -> tree

        Returns:
            Self for method chaining.
        """
        condition = var(EgglogConstants.VAR_CONDITION, Condition)
        leaf = var(EgglogConstants.VAR_LEAF, Tree)

        self._rules.append(rewrite(Tree.if_(condition, leaf, leaf)).to(leaf))
        return self

    def add_dead_branch_pruning(self) -> RuleSet:
        """Add dead branch pruning rules.

        Eliminates duplicate conditions along a path where the outcome is already determined:
        - If c is True: If(c, If(c, a, b), d) -> If(c, a, d)  (b is unreachable)
        - If c is False: If(c, d, If(c, a, b)) -> If(c, d, b)  (a is unreachable)

        Returns:
            Self for method chaining.
        """
        condition = var(EgglogConstants.VAR_CONDITION, Condition)
        then_branch = var(EgglogConstants.VAR_THEN_BRANCH, Tree)
        else_branch = var(EgglogConstants.VAR_ELSE_BRANCH, Tree)
        leaf = var(EgglogConstants.VAR_LEAF, Tree)

        # Rule 1: Inner then-branch redundant
        self._rules.append(
            rewrite(
                Tree.if_(
                    condition,
                    Tree.if_(condition, then_branch, else_branch),
                    leaf,
                )
            ).to(Tree.if_(condition, then_branch, leaf))
        )

        # Rule 2: Inner else-branch contradictory
        self._rules.append(
            rewrite(
                Tree.if_(
                    condition,
                    leaf,
                    Tree.if_(condition, then_branch, else_branch),
                )
            ).to(Tree.if_(condition, leaf, else_branch))
        )

        return self

    def add_subtree_replacement(self) -> RuleSet:
        """Add the subtree replacement rule.

        This rule replaces subtrees where all leaf paths lead to the same value.
        If all leaves in a subtree have the same value, replace with a single leaf:
        If(c1, If(c2, leaf(v), leaf(v)), leaf(v)) -> leaf(v)

        Returns:
            Self for method chaining.
        """
        condition1 = var(EgglogConstants.VAR_CONDITION1, Condition)
        condition2 = var(EgglogConstants.VAR_CONDITION2, Condition)
        leaf_value = var(EgglogConstants.VAR_LEAF_VALUE, f64Like)  # type: ignore[var-annotated, arg-type]
        leaf = Tree.leaf(leaf_value)

        # Rule 1: Then branch has identical leaves
        self._rules.append(
            rewrite(
                Tree.if_(
                    condition1,
                    Tree.if_(condition2, leaf, leaf),
                    leaf,
                )
            ).to(leaf)
        )

        # Rule 2: Else branch has identical leaves
        self._rules.append(
            rewrite(
                Tree.if_(
                    condition1,
                    leaf,
                    Tree.if_(condition2, leaf, leaf),
                )
            ).to(leaf)
        )

        return self

    def add_subtree_raising(self) -> RuleSet:
        """Add subtree raising rules.

        Promotes a child condition above its parent when the child is on a different feature.
        Note: Can significantly increase e-graph saturation time.

        Returns:
            Self for method chaining.
        """
        feature1 = var(EgglogConstants.VAR_FEATURE1, StringLike)  # type: ignore[var-annotated, arg-type]
        feature2 = var(EgglogConstants.VAR_FEATURE2, StringLike)  # type: ignore[var-annotated, arg-type]
        threshold1 = var(EgglogConstants.VAR_THRESHOLD1, f64Like)  # type: ignore[var-annotated, arg-type]
        threshold2 = var(EgglogConstants.VAR_THRESHOLD2, f64Like)  # type: ignore[var-annotated, arg-type]
        branch_left = var(EgglogConstants.VAR_BRANCH_LEFT, Tree)
        branch_middle = var(EgglogConstants.VAR_BRANCH_MIDDLE, Tree)
        branch_right = var(EgglogConstants.VAR_BRANCH_RIGHT, Tree)

        self._rules.append(
            rewrite(
                Tree.if_(
                    Condition(feature1, threshold1),
                    Tree.if_(Condition(feature2, threshold2), branch_left, branch_middle),
                    branch_right,
                )
            ).to(
                Tree.if_(
                    Condition(feature2, threshold2),
                    Tree.if_(Condition(feature1, threshold1), branch_left, branch_right),
                    Tree.if_(Condition(feature1, threshold1), branch_middle, branch_right),
                )
            )
        )

        return self

    def __len__(self) -> int:
        """Return the number of rules in the set."""
        return len(self._rules)

    def __iter__(self):  # type: ignore[no-untyped-def]
        """Iterate over the rules."""
        return iter(self._rules)

    def __repr__(self) -> str:
        """Return string representation of the rule set."""
        return f"RuleSet(name='{self.name}', rules={len(self._rules)})"

    @classmethod
    def create(cls, name: Literal["basic", "full"]) -> RuleSet:
        """Create a named rule set.

        Args:
            name: Name of the rule set to create:
                - Constants.RULE_SET_BASIC: Basic redundant split elimination
                - Constants.RULE_SET_FULL: All available sound simplification rules

        Returns:
            A configured RuleSet instance.

        Raises:
            ValueError: If an unknown rule set name is provided.
        """
        if name == Constants.RULE_SET_BASIC:
            return cls.basic()
        elif name == Constants.RULE_SET_FULL:
            return cls.full()
        else:
            raise ValueError(
                f"Unknown rule set: {name}. Use '{Constants.RULE_SET_BASIC}' "
                f"or '{Constants.RULE_SET_FULL}'."
            )

    @classmethod
    def basic(cls) -> RuleSet:
        """Create the basic rule set with redundant split elimination.

        Returns:
            RuleSet with basic simplification rules.
        """
        rules = cls(name=Constants.RULE_SET_BASIC)
        rules.add_redundant_split_elimination()
        return rules

    @classmethod
    def full(cls) -> RuleSet:
        """Create the full rule set with sound simplification rules.

        Includes:
        - Redundant split elimination
        - Dead branch pruning (identical condition elimination)
        - Subtree replacement (homogeneous leaf collapsing)

        Returns:
            RuleSet with all sound simplification rules.
        """
        rules = cls(name=Constants.RULE_SET_FULL)
        rules.add_redundant_split_elimination()
        rules.add_dead_branch_pruning()
        rules.add_subtree_replacement()
        return rules
