"""Egglog Tree expression type for representing decision trees."""

from __future__ import annotations

from egglog import *

from .condition import Condition


class Tree(Expr, egg_sort="Tree"):
    """Decision tree expression type.

    Trees can be constructed as:
    - Leaf nodes: Tree.leaf(value) containing a prediction value
    - If nodes: Tree.if_(condition, then_branch, else_branch) for splits

    For classification, the value is the class index (stored as float).
    For regression, the value is the predicted continuous value.
    """

    @method(egg_fn="Leaf")
    @classmethod
    def leaf(cls, value: f64Like) -> Tree: ...

    @method(egg_fn="If")
    @classmethod
    def if_(cls, condition: Condition, then_branch: Tree, else_branch: Tree) -> Tree: ...

    def value(self) -> f64: ...
    def condition(self) -> Condition: ...
    def then_branch(self) -> Tree: ...
    def else_branch(self) -> Tree: ...
