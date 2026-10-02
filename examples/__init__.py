"""Example modules for TreE-Graph visualization and metrics.

This package provides:
- MetricsCalculator: Compute and compare classification metrics
- TreeVisualizer: Create visualizations comparing decision trees
"""

from examples.dataset_loader import get_supported_datasets
from examples.dataset_loader import load_dataset
from examples.log_loss_result import LogLossResult
from examples.metrics_calculator import MetricsCalculator
from examples.roc_result import ROCResult
from examples.tree_visualizer import TreeVisualizer

__all__ = [
    "LogLossResult",
    "MetricsCalculator",
    "ROCResult",
    "TreeVisualizer",
    "get_supported_datasets",
    "load_dataset",
]
