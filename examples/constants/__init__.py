"""Constants package for tree comparison plots and examples.

This module re-exports the Constants class for backward compatibility.
The Constants class inherits from all sub-module classes to provide a
single access point for all constants.

Usage::

    from examples.constants import Constants

    color = Constants.COLOR_ORIGINAL
    zero = Constants.ZERO
    label = Constants.LABEL_ORIGINAL_TREE_PREDICTION
    test_size = Constants.TEST_SIZE
"""

from examples.constants.number_constants import NumberConstants as NumberConstants
from examples.constants.string_constants import StringConstants as StringConstants
from examples.constants.training_constants import TrainingConstants as TrainingConstants
from examples.constants.visualization_constants import (
    VisualizationConstants as VisualizationConstants,
)


class Constants(VisualizationConstants, NumberConstants, StringConstants, TrainingConstants):
    """Composite constants class for backward compatibility.

    Inherits from all sub-module constant classes to provide a single
    access point for all visualization, numeric, string, and training
    constants.
    """
