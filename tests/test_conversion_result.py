"""Tests for ConversionResult dataclass."""

from egglog import EGraph
from egglog import f64

from tests.constants import ONE_FLOAT
from tests.constants import ZERO_FLOAT
from tree_graph import ConversionResult
from tree_graph import Tree


class TestConversionResult:
    """Tests for the ConversionResult class."""

    def test_creation(self) -> None:
        """Test creating a ConversionResult."""
        egraph = EGraph()
        root_term = Tree.leaf(f64(ZERO_FLOAT))

        result = ConversionResult(egraph, root_term)

        assert result.egraph is egraph
        assert result.root_term is root_term

    def test_dataclass_fields(self) -> None:
        """Test that all fields are accessible."""
        egraph = EGraph()
        root_term = Tree.leaf(f64(ONE_FLOAT))

        result = ConversionResult(egraph, root_term)

        # All fields should be accessible
        assert hasattr(result, "egraph")
        assert hasattr(result, "root_term")
