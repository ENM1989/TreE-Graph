"""Dataclass result test constants."""


class TestDataclassConstants:
    """Constants for testing dataclass result objects."""

    # SimplificationResult test values
    ORIGINAL_NODES_TEN: int = 10
    SIMPLIFIED_NODES_SEVEN: int = 7
    REDUCTION_PERCENT_THIRTY: float = 30.0
    ITERATIONS_RUN_HUNDRED: int = 100

    # SimplificationResult alternate test values
    ORIGINAL_NODES_HUNDRED: int = 100
    SIMPLIFIED_NODES_EIGHTY: int = 80
    REDUCTION_PERCENT_TWENTY: float = 20.0
    ITERATIONS_RUN_FIFTY: int = 50

    # ForestSimplificationResult test values
    FOREST_ORIGINAL_TOTAL_NODES_HUNDRED: int = 100
    FOREST_SIMPLIFIED_TOTAL_NODES_EIGHTY: int = 80
    FOREST_ORIGINAL_AVG_NODES_TEN: float = 10.0
    FOREST_SIMPLIFIED_AVG_NODES_EIGHT: float = 8.0
    FOREST_TREES_SIMPLIFIED_TEN: int = 10

    # ForestSimplificationResult alternate test values
    FOREST_ORIGINAL_TOTAL_NODES_TWO_HUNDRED: int = 200
    FOREST_SIMPLIFIED_TOTAL_NODES_ONE_FIFTY: int = 150
    FOREST_ORIGINAL_AVG_NODES_TWENTY: float = 20.0
    FOREST_SIMPLIFIED_AVG_NODES_FIFTEEN: float = 15.0
    FOREST_REDUCTION_PERCENT_TWENTY_FIVE: float = 25.0
