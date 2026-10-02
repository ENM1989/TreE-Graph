"""Result type for parsed condition."""

from dataclasses import dataclass


@dataclass
class ParsedCondition:
    """Parsed condition from an If term string.

    Attributes:
        feature_idx: Index of the feature used in the condition.
        threshold: Threshold value for the split.
        then_term: String representation of the then branch.
        else_term: String representation of the else branch.
    """

    feature_idx: int
    threshold: float
    then_term: str
    else_term: str
