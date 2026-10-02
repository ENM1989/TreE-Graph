"""Result type for tree simplification operations."""

from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


@dataclass
class SimplificationResult:
    """Result of tree simplification operation.

    Attributes:
        estimator: The simplified decision tree estimator.
        original_nodes: Number of nodes in original tree.
        simplified_nodes: Number of nodes in simplified tree.
        reduction_percent: Percentage reduction in nodes.
        iterations_run: Number of e-graph iterations performed.
        predictions_preserved: Whether predictions match.
    """

    estimator: DecisionTreeClassifier | DecisionTreeRegressor
    original_nodes: int
    simplified_nodes: int
    reduction_percent: float
    iterations_run: int
    predictions_preserved: bool
