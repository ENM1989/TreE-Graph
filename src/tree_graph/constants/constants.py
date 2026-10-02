"""Library-wide constants.

This module defines general-purpose constants used across the tree-graph library,
including version information, default parameters, and shared string values.
"""

from typing import Literal


class Constants:
    """Library-wide constants.

    Attributes:
        VERSION: Library version string.
        DEFAULT_MAX_ITERATIONS: Default maximum e-graph saturation iterations.
    """

    # Version
    VERSION: str = "0.1.0"

    # Default iteration counts
    DEFAULT_MAX_ITERATIONS: int = 100

    # Warning messages
    SIMPLIFICATION_PREDICTION_MISMATCH_MESSAGE: str = (
        "Simplification produced different predictions. Returning original estimator."
    )

    # E-graph expression strings
    EGRAPH_LEAF_EXPRESSION: str = "Tree.leaf(0)"
    EGRAPH_TREE_PREFIX: str = "Tree."
    EXTRACTOR_PREFIX_LENGTH: int = 5

    # Rule set names
    RULE_SET_BASIC: Literal["basic"] = "basic"
    RULE_SET_FULL: Literal["full"] = "full"
