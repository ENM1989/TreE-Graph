"""Tests for dataset loading utilities."""

import numpy as np
import pytest
from sklearn.datasets import load_iris

from examples.dataset_loader import get_supported_datasets
from examples.dataset_loader import load_dataset


class TestDatasetLoader:
    """Tests for the load_dataset function and supported datasets."""

    def test_get_supported_datasets(self) -> None:
        """Test retrieving supported dataset names."""
        datasets = get_supported_datasets()
        assert "iris" in datasets
        assert "wine" in datasets
        assert "breast_cancer" in datasets
        assert "digits" in datasets
        assert "moons" in datasets
        assert "blobs" in datasets
        assert "classification" in datasets

    def test_load_iris_default(self) -> None:
        """Test loading default dataset (iris)."""
        features, labels, feature_names, class_names = load_dataset()
        assert features.shape == (150, 4)
        assert labels.shape == (150,)
        assert len(feature_names) == 4
        assert len(class_names) == 3
        assert class_names == ["setosa", "versicolor", "virginica"]

    def test_load_wine(self) -> None:
        """Test loading wine dataset."""
        features, labels, feature_names, class_names = load_dataset("wine")
        assert features.shape == (178, 13)
        assert labels.shape == (178,)
        assert len(feature_names) == 13
        assert len(class_names) == 3

    def test_load_breast_cancer(self) -> None:
        """Test loading breast cancer dataset and aliases."""
        features, labels, feature_names, class_names = load_dataset("breast_cancer")
        assert features.shape == (569, 30)
        assert labels.shape == (569,)
        assert len(feature_names) == 30
        assert len(class_names) == 2

        # Test alias "cancer"
        features_alias, labels_alias, _, _ = load_dataset("cancer")
        assert features_alias.shape == features.shape
        assert labels_alias.shape == labels.shape

    def test_load_digits(self) -> None:
        """Test loading digits dataset with string converted class names."""
        features, labels, feature_names, class_names = load_dataset("digits")
        assert features.shape == (1797, 64)
        assert labels.shape == (1797,)
        assert len(feature_names) == 64
        assert len(class_names) == 10
        assert all(isinstance(c, str) for c in class_names)
        assert class_names == ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def test_load_titanic(self) -> None:
        """Test loading Titanic dataset with preprocessing."""
        features, labels, feature_names, class_names = load_dataset("titanic")
        assert features.shape == (1309, 7)
        assert labels.shape == (1309,)
        assert feature_names == ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
        assert class_names == ["perished", "survived"]
        assert not np.isnan(features).any()

    def test_load_synthetic_moons(self) -> None:
        """Test loading synthetic make_moons dataset."""
        features, labels, feature_names, class_names = load_dataset("moons")
        assert features.shape == (300, 2)
        assert labels.shape == (300,)
        assert len(feature_names) == 2
        assert len(class_names) == 2

    def test_load_synthetic_blobs(self) -> None:
        """Test loading synthetic make_blobs dataset."""
        features, labels, feature_names, class_names = load_dataset("blobs")
        assert features.shape == (300, 4)
        assert labels.shape == (300,)
        assert len(feature_names) == 4
        assert len(class_names) == 3

    def test_load_synthetic_classification(self) -> None:
        """Test loading synthetic make_classification dataset."""
        features, labels, feature_names, class_names = load_dataset("classification")
        assert features.shape == (300, 10)
        assert labels.shape == (300,)
        assert len(feature_names) == 10
        assert len(class_names) == 3

    def test_load_bunch_directly(self) -> None:
        """Test passing an existing sklearn Bunch directly."""
        bunch = load_iris()
        features, labels, feature_names, class_names = load_dataset(bunch)
        assert features.shape == (150, 4)
        assert labels.shape == (150,)
        assert len(feature_names) == 4
        assert len(class_names) == 3

    def test_load_callable(self) -> None:
        """Test passing a loader callable."""
        features, labels, feature_names, class_names = load_dataset(load_iris)
        assert features.shape == (150, 4)
        assert labels.shape == (150,)
        assert len(feature_names) == 4
        assert len(class_names) == 3

    def test_load_tuple(self) -> None:
        """Test passing a 2-tuple and 4-tuple."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])

        features, labels, feature_names, class_names = load_dataset((X, y))
        assert features.shape == (3, 2)
        assert labels.shape == (3,)
        assert len(feature_names) == 2
        assert len(class_names) == 2

        f_names = ["alpha", "beta"]
        c_names = ["neg", "pos"]
        features, labels, feature_names, class_names = load_dataset((X, y, f_names, c_names))
        assert feature_names == f_names
        assert class_names == c_names

    def test_load_non_contiguous_labels_encoded(self) -> None:
        """Test that non-contiguous labels are safely encoded to 0..K-1."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([10, 25, 10, 25])

        features, labels, feature_names, class_names = load_dataset((X, y))
        assert np.array_equal(labels, np.array([0, 1, 0, 1]))
        assert class_names == ["10", "25"]

    def test_load_string_labels_encoded(self) -> None:
        """Test that string labels are safely encoded to integers."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array(["cat", "dog", "cat"])

        features, labels, feature_names, class_names = load_dataset((X, y))
        assert np.array_equal(labels, np.array([0, 1, 0]))
        assert class_names == ["cat", "dog"]

    def test_load_unknown_dataset_raises(self) -> None:
        """Test that an unknown dataset name raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown dataset 'nonexistent_dataset'"):
            load_dataset("nonexistent_dataset")

    def test_load_invalid_tuple_length_raises(self) -> None:
        """Test that a tuple with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="Dataset tuple must have length 2"):
            load_dataset((np.array([1]), np.array([1]), np.array([1])))

    def test_load_unsupported_type_raises(self) -> None:
        """Test that an unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            load_dataset(12345)
