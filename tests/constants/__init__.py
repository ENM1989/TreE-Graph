"""Constants used across test files.

This module defines all magic numbers and strings as named constants
for better maintainability and clarity in tests.

Classes and module-level constants are re-exported here for convenience
so that existing imports continue to work.
"""

from tests.constants.pytest_constants import PytestConstants as PytestConstants
from tests.constants.test_constants import TestConstants as TestConstants
from tests.constants.test_dataclass_constants import (
    TestDataclassConstants as TestDataclassConstants,
)
from tests.constants.test_edge_case_constants import TestEdgeCaseConstants as TestEdgeCaseConstants
from tests.constants.test_feature_constants import TestFeatureConstants as TestFeatureConstants
from tests.constants.test_node_constants import TestNodeConstants as TestNodeConstants
from tree_graph.constants import Constants as Constants
from tree_graph.constants import NumberConstants as NumberConstants

# Re-export rule set constants from tree_graph for convenience
RULE_SET_BASIC = Constants.RULE_SET_BASIC
RULE_SET_FULL = Constants.RULE_SET_FULL

# Test-specific rule set value (module-level for convenience)
RULE_SET_INVALID: str = TestEdgeCaseConstants.INVALID_THRESHOLD_TEXT

# Number constants for tests
ZERO = NumberConstants.ZERO
ONE = NumberConstants.ONE
TWO = NumberConstants.TWO
FIVE = NumberConstants.FIVE
MINUS_TWO = NumberConstants.MINUS_TWO

# Float constants for tests
ZERO_FLOAT = NumberConstants.ZERO_FLOAT
ONE_FLOAT = NumberConstants.ONE_FLOAT
HALF = NumberConstants.HALF

# Re-export commonly used test constants at module level for convenience
RANDOM_SEED = TestConstants.RANDOM_SEED
MAX_DEPTH_ONE = TestConstants.MAX_DEPTH_ONE
MAX_DEPTH_TWO = TestConstants.MAX_DEPTH_TWO
MAX_DEPTH_THREE = TestConstants.MAX_DEPTH_THREE
MAX_DEPTH_FOUR = TestConstants.MAX_DEPTH_FOUR
MAX_DEPTH_FIVE = TestConstants.MAX_DEPTH_FIVE
MAX_DEPTH_TEN = TestConstants.MAX_DEPTH_TEN
N_ESTIMATORS_TWO = TestConstants.N_ESTIMATORS_TWO
N_ESTIMATORS_THREE = TestConstants.N_ESTIMATORS_THREE
N_ESTIMATORS_FIVE = TestConstants.N_ESTIMATORS_FIVE
N_ESTIMATORS_TEN = TestConstants.N_ESTIMATORS_TEN
MAX_ITERATIONS_ONE = TestConstants.MAX_ITERATIONS_ONE
MAX_ITERATIONS_FIVE = TestConstants.MAX_ITERATIONS_FIVE
MAX_ITERATIONS_TEN = TestConstants.MAX_ITERATIONS_TEN
MAX_ITERATIONS_TWENTY = TestConstants.MAX_ITERATIONS_TWENTY
MAX_ITERATIONS_FIFTY = TestConstants.MAX_ITERATIONS_FIFTY
MAX_ITERATIONS_HUNDRED = TestConstants.MAX_ITERATIONS_HUNDRED
TEST_SIZE_TWENTY_PERCENT = TestConstants.TEST_SIZE_TWENTY_PERCENT
TEST_SIZE_THIRTY_PERCENT = TestConstants.TEST_SIZE_THIRTY_PERCENT
N_SAMPLES_HUNDRED = TestConstants.N_SAMPLES_HUNDRED
N_SAMPLES_TWO_HUNDRED = TestConstants.N_SAMPLES_TWO_HUNDRED
N_FEATURES_FOUR = TestConstants.N_FEATURES_FOUR
GRID_SIZE_TEN = TestConstants.GRID_SIZE_TEN
