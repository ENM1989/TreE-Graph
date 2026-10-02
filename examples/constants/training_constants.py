"""Training and model selection constants for examples.

This module provides the TrainingConstants class with training/test split
parameters, GridSearchCV parameters, and print formatting constants used
consistently across example scripts.
"""


class TrainingConstants:
    """Training and model selection constants for examples.

    This class contains training/test split parameters, GridSearchCV
    parameters, and print formatting constants used across the project
    for consistent training and evaluation settings.

    Attributes:
        GRID_MAX_DEPTH_VALUES: Max depth values for grid search.
        GRID_MIN_SAMPLES_SPLIT_VALUES: Min samples split values for grid search.
        GRID_MIN_SAMPLES_LEAF_VALUES: Min samples leaf values for grid search.
        TEST_SIZE: Test set proportion.
        RANDOM_STATE: Random state for reproducibility.
        K_FOLDS: Number of cross-validation folds.
        MAX_ITERATIONS: Maximum iterations for simplification.
        HEADER_WIDTH_70: Print header width 70.
        HEADER_WIDTH_45: Print header width 45.
        HEADER_WIDTH_57: Print header width 57.
        HEADER_WIDTH_60: Print header width 60.
        PRINT_EQUALS: Equals separator for print output.
    """

    # GridSearchCV parameters
    GRID_MAX_DEPTH_VALUES: list[int | None] = [2, 3, 5, 10, None]
    GRID_MIN_SAMPLES_SPLIT_VALUES: list[int] = [2, 5, 10]
    GRID_MIN_SAMPLES_LEAF_VALUES: list[int] = [1, 2, 5]

    # Training/test split
    TEST_SIZE: float = 0.3
    RANDOM_STATE: int = 42
    K_FOLDS: int = 5
    MAX_ITERATIONS: int = 100

    # Print formatting
    HEADER_WIDTH_70: int = 70
    HEADER_WIDTH_45: int = 45
    HEADER_WIDTH_57: int = 57
    HEADER_WIDTH_60: int = 60
    PRINT_EQUALS: str = "="
