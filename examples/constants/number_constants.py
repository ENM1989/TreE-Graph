"""Numeric constants for tree comparison plots and examples.

This module provides the NumberConstants class with integer, float, and
boolean constants used consistently across all visualization and training
functions.
"""

from typing import Literal


class NumberConstants:
    """Numeric constants for tree comparison plots and examples.

    This class contains integer, float, and boolean constants used across
    the project for consistent numeric values.

    Attributes:
        ZERO: Integer zero.
        ONE: Integer one.
        TWO: Integer two.
        THREE: Integer three.
        FOUR: Integer four.
        FIVE: Integer five.
        SIX: Integer six.
        EIGHT: Integer eight.
        TEN: Integer ten.
        TWENTY: Integer twenty.
        HUNDRED: Integer hundred.
        MINUS_ONE: Integer minus one.
        HALF: Float 0.5.
        MINUS_HALF: Float -0.5.
        TENTH: Float 0.1.
        ZERO_FLOAT: Float 0.0.
        ONE_FLOAT: Float 1.0.
        ONE_POINT_ZERO_FIVE: Float 1.05.
        COMPARE_EPSILON: Small epsilon for comparisons.
        LOG_LOSS_COMPARE_EPSILON: Small epsilon for log loss comparisons.
        PLOT_OFFSET: Offset for plot elements.
        PLOT_RANGE_MIN: Minimum plot range value.
        PLOT_RANGE_MAX: Maximum plot range value.
        PLOT_RANGE_PAD: Padding for plot range.
        ALPHA_HALF: Alpha value 0.5.
        ALPHA_0_3: Alpha value 0.3.
        ASPECT_EQUAL: Literal "equal" for aspect ratio.
        FIGURE_WIDTH_10: Figure width 10.
        FIGURE_WIDTH_14: Figure width 14.
        FIGURE_WIDTH_20: Figure width 20.
        FIGURE_HEIGHT_6: Figure height 6.
        FIGURE_HEIGHT_8: Figure height 8.
        FIGURE_SCALE_5: Figure scale 5.
        FIGURE_SCALE_4: Figure scale 4.
        RANDOM_SEED_JITTER: Random seed for jitter.
        MARKER_SIZE_10: Marker size 10.
        LEGEND_FONTSIZE_8: Legend fontsize 8.
        TITLE_FONTSIZE_16: Title fontsize 16.
        LINE_WIDTH_1: Line width 1.
        LINE_WIDTH_2: Line width 2.
        TRUE: Boolean True constant.
        FALSE: Boolean False constant.
        LINEWIDTH_HALF: Line width 0.5.
    """

    # Integer constants
    ZERO: int = 0
    ONE: int = 1
    TWO: int = 2
    THREE: int = 3
    FOUR: int = 4
    FIVE: int = 5
    SIX: int = 6
    EIGHT: int = 8
    TEN: int = 10
    TWENTY: int = 20
    HUNDRED: int = 100
    MINUS_ONE: int = -1

    # Float constants
    HALF: float = 0.5
    MINUS_HALF: float = -0.5
    TENTH: float = 0.1
    ZERO_FLOAT: float = 0.0
    ONE_FLOAT: float = 1.0
    ONE_POINT_ZERO_FIVE: float = 1.05
    COMPARE_EPSILON: float = 1e-6
    LOG_LOSS_COMPARE_EPSILON: float = 1e-6
    PLOT_OFFSET: float = 0.02
    PLOT_RANGE_MIN: float = 0.01
    PLOT_RANGE_MAX: float = 0.99
    PLOT_RANGE_PAD: float = 0.1
    ALPHA_HALF: float = 0.5
    ALPHA_0_3: float = 0.3
    ASPECT_EQUAL: Literal["equal"] = "equal"
    FIGURE_WIDTH_10: int = 10
    FIGURE_WIDTH_14: int = 14
    FIGURE_WIDTH_20: int = 20
    FIGURE_HEIGHT_6: int = 6
    FIGURE_HEIGHT_8: int = 8
    FIGURE_SCALE_5: int = 5
    FIGURE_SCALE_4: int = 4
    RANDOM_SEED_JITTER: int = 43
    MARKER_SIZE_10: int = 10
    LEGEND_FONTSIZE_8: int = 8
    TITLE_FONTSIZE_16: int = 16
    LINE_WIDTH_1: int = 1
    LINE_WIDTH_2: int = 2
    LINEWIDTH_HALF: float = 0.5

    # Boolean constants
    TRUE: bool = True
    FALSE: bool = False

    # Probability range for plots
    PROBABILITY_MIN: float = 0.01
    PROBABILITY_MAX: float = 0.99
