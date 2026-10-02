"""Test configuration constants.

This module defines test-specific constants including random seeds,
test parameters, and e-graph iteration counts.
"""


class TestConstants:
    """Test-specific constants.

    These are values used specifically for testing, including
    random seeds, test parameters, and test data values.
    """

    # Random seeds for reproducibility
    RANDOM_SEED: int = 42

    # Tree depth values for testing
    MAX_DEPTH_ONE: int = 1
    MAX_DEPTH_TWO: int = 2
    MAX_DEPTH_THREE: int = 3
    MAX_DEPTH_FOUR: int = 4
    MAX_DEPTH_FIVE: int = 5
    MAX_DEPTH_TEN: int = 10

    # Number of estimators for ensemble tests
    N_ESTIMATORS_TWO: int = 2
    N_ESTIMATORS_THREE: int = 3
    N_ESTIMATORS_FIVE: int = 5
    N_ESTIMATORS_TEN: int = 10

    # E-graph iteration counts
    MAX_ITERATIONS_ONE: int = 1
    MAX_ITERATIONS_FIVE: int = 5
    MAX_ITERATIONS_TEN: int = 10
    MAX_ITERATIONS_TWENTY: int = 20
    MAX_ITERATIONS_FIFTY: int = 50
    MAX_ITERATIONS_HUNDRED: int = 100

    # Train/test split ratios
    TEST_SIZE_TWENTY_PERCENT: float = 0.2
    TEST_SIZE_THIRTY_PERCENT: float = 0.3

    # Sample counts for test data
    N_SAMPLES_HUNDRED: int = 100
    N_SAMPLES_TWO_HUNDRED: int = 200

    # Feature counts
    N_FEATURES_FOUR: int = 4

    # Grid sizes for synthetic test data
    GRID_SIZE_TEN: int = 10
