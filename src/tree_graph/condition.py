"""Egglog Condition type for decision tree split conditions."""

from __future__ import annotations

from egglog import *


class Condition(Expr):
    """Split condition: feature < threshold.

    Attributes:
        feature: Name of the feature to split on.
        threshold: Threshold value for the split.
    """

    @method(egg_fn="Condition")
    def __init__(self, feature: StringLike, threshold: f64Like) -> None: ...

    def feature(self) -> String: ...
    def threshold(self) -> f64: ...
