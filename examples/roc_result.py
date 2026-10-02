"""Result type for ROC curve visualization."""

from dataclasses import dataclass


@dataclass
class ROCResult:
    """Result of ROC curve plot operation.

    Attributes:
        save_path: Path where the figure was saved.
        auc_scores: Dictionary mapping 'original' and 'simplified' to
            class name -> AUC score mappings.
    """

    save_path: str
    auc_scores: dict[str, dict[str, float]]
