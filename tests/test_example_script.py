"""Tests for the decision tree example script."""

from examples.constants import Constants
from examples.decision_tree import parse_args


class TestExampleScript:
    """Tests for example script arguments and functions."""

    def test_parse_args_defaults(self) -> None:
        """Test default argument values."""
        args = parse_args([])
        assert args.dataset == Constants.DEFAULT_DATASET
        assert args.output_dir == Constants.DEFAULT_OUTPUT_DIR
        assert args.output_prefix is None

    def test_parse_args_custom(self) -> None:
        """Test custom CLI arguments."""
        args = parse_args(["-d", "wine", "-o", "custom_dir", "--output-prefix", "wine_test_"])
        assert args.dataset == "wine"
        assert args.output_dir == "custom_dir"
        assert args.output_prefix == "wine_test_"

    def test_parse_args_long_flag(self) -> None:
        """Test parsing with --dataset."""
        args = parse_args(["--dataset", "breast_cancer"])
        assert args.dataset == "breast_cancer"
