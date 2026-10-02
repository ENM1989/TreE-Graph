"""Visualization constants for tree comparison plots.

This module provides the VisualizationConstants class with color schemes,
styling parameters, and figure settings used consistently across all
visualization functions.
"""


class VisualizationConstants:
    """Visualization constants for tree comparison plots.

    This class contains color schemes, styling parameters, and figure
    settings used in the TreeVisualizer for consistent visualization output.

    Attributes:
        COLOR_ORIGINAL: Color for original model data points.
        COLOR_SIMPLIFIED: Color for simplified model data points.
        COLOR_AGREE_CORRECT: Color for points where both models agree and are correct.
        COLOR_AGREE_WRONG: Color for points where both models agree but are wrong.
        COLOR_BOTH_WRONG_DIFF: Color for points where both models are wrong differently.
        COLOR_EQUAL: Color for points with equal performance.
        CLASS_COLORS: List of colors for multi-class plots.
        DEFAULT_ALPHA: Default transparency for scatter plots.
        DEFAULT_POINT_SIZE: Default point size for scatter plots.
        DEFAULT_SMALL_POINT_SIZE: Smaller point size for dense plots.
        DEFAULT_LINE_WIDTH: Default line width for plot elements.
        DEFAULT_THICK_LINE_WIDTH: Thicker line width for emphasis.
        DEFAULT_MARKER_EDGE_WIDTH: Default marker edge width.
        FONTSIZE_TITLE: Font size for plot titles.
        FONTSIZE_LABEL: Font size for axis labels.
        FONTSIZE_TICK: Font size for tick labels.
        FONTSIZE_LEGEND: Font size for legend text.
        FONTSIZE_SMALL: Small font size for subtitles.
        DEFAULT_DPI: Default dots per inch for saved figures.
        DEFAULT_BBOX_INCHES: Default bounding box setting for saved figures.
        SCATTER_JITTER: Jitter amount for scatter plots.
        LOG_LOSS_JITTER: Jitter amount for log loss scatter plots.
        LOG_EPSILON: Small value to prevent log(0) errors.
        GRID_ALPHA: Alpha transparency for grid lines.
    """

    # Color scheme (consistent across all plots)
    # Original model = Blue, Simplified model = Red
    COLOR_ORIGINAL: str = "#1f77b4"
    COLOR_SIMPLIFIED: str = "#d62728"
    COLOR_AGREE_CORRECT: str = "#2ca02c"  # Both agree and correct
    COLOR_AGREE_WRONG: str = "#7f7f7f"  # Both agree but wrong
    COLOR_BOTH_WRONG_DIFF: str = "#9467bd"  # Both wrong, different predictions
    COLOR_EQUAL: str = "#7f7f7f"  # Equal performance

    # Matplotlib default color cycle for multi-class plots
    CLASS_COLORS: list[str] = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
    ]

    # Plot styling defaults
    DEFAULT_ALPHA: float = 0.6
    DEFAULT_POINT_SIZE: int = 80
    DEFAULT_SMALL_POINT_SIZE: int = 60
    DEFAULT_LINE_WIDTH: float = 0.5
    DEFAULT_THICK_LINE_WIDTH: int = 2
    DEFAULT_MARKER_EDGE_WIDTH: float = 0.5

    # Font sizes
    FONTSIZE_TITLE: int = 14
    FONTSIZE_LABEL: int = 12
    FONTSIZE_TICK: int = 11
    FONTSIZE_LEGEND: int = 9
    FONTSIZE_SMALL: int = 10

    # Figure settings
    DEFAULT_DPI: int = 150
    DEFAULT_BBOX_INCHES: str = "tight"

    # Jitter amounts for scatter plots
    SCATTER_JITTER: float = 0.08
    LOG_LOSS_JITTER: float = 0.02

    # Numerical constants
    LOG_EPSILON: float = 1e-15

    # Grid settings
    GRID_ALPHA: float = 0.3

    # Matplotlib backend
    MATPLOTLIB_BACKEND_AGG: str = "Agg"
