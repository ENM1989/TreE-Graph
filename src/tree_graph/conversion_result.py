"""Result type for tree to e-graph conversion."""

from dataclasses import dataclass

from egglog import EGraph

from .tree import Tree


@dataclass
class ConversionResult:
    """Result of converting a decision tree to an e-graph.

    Attributes:
        egraph: EGraph instance with tree structure encoded.
        root_term: Root Tree term representing the tree structure.
    """

    egraph: EGraph
    root_term: Tree
