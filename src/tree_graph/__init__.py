"""TreE-Graph: Simplify scikit-learn decision trees using egglog e-graphs."""

from .condition import Condition
from .constants import Constants
from .conversion_result import ConversionResult
from .parsed_condition import ParsedCondition
from .rule_set import RuleSet
from .simplification_result import SimplificationResult
from .tree import Tree
from .tree_converter import TreeConverter
from .tree_extractor import TreeExtractor
from .tree_simplifier import TreeSimplifier
from .tree_simplifier import simplify_tree
from .tree_utils import TreeUtils

__version__ = Constants.VERSION

__all__ = [
    "Condition",
    "Constants",
    "ConversionResult",
    "ParsedCondition",
    "RuleSet",
    "SimplificationResult",
    "Tree",
    "TreeConverter",
    "TreeExtractor",
    "TreeSimplifier",
    "TreeUtils",
    "simplify_tree",
    "__version__",
]
