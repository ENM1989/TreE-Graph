"""Result type for log loss visualization."""

from dataclasses import dataclass


@dataclass
class LogLossResult:
    """Result of log loss scatter plot operation.

    Attributes:
        save_path: Path where the figure was saved.
        original_log_loss: Log loss of the original model.
        simplified_log_loss: Log loss of the simplified model.
    """

    save_path: str
    original_log_loss: float
    simplified_log_loss: float
