"""Edge case test constants."""


class TestEdgeCaseConstants:
    """Constants for edge case tests."""

    # Invalid node index for testing
    INVALID_NODE_INDEX: int = 999

    # Gini test values
    GINI_PURE_VALUE_1: float = 10.0
    GINI_MIXED_VALUE_1: float = 5.0

    # Invalid strings for parsing tests
    INVALID_THRESHOLD_TEXT: str = "invalid"
    INVALID_LEAF_STRING: str = "Tree.leaf(invalid)"
    EMPTY_LEAF_STRING: str = "Tree.leaf()"
    INVALID_IF_STRING: str = 'Tree.if_(Condition("feature_0", invalid), Tree.leaf(0), Tree.leaf(1))'
    UNRECOGNIZED_TERM_STRING: str = "UnrecognizedTerm()"
