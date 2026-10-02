"""Dataset loading utilities for TreE-Graph examples.

This module provides functions to load and prepare various scikit-learn
datasets for use in example scripts and tree simplification workflows.
"""

from collections.abc import Callable
from typing import TypeAlias

from numpy import asarray
from numpy import integer
from numpy import issubdtype
from numpy import ndarray
from numpy import unique
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from sklearn.utils import Bunch

from examples.constants import Constants

DatasetInput: TypeAlias = (
    str
    | Bunch
    | Callable[..., object]
    | tuple[ndarray, ndarray]
    | tuple[ndarray, ndarray, list[str], list[str]]
    | object
)

SUPPORTED_DATASETS: dict[str, str] = {
    Constants.DATASET_IRIS: "Iris flower classification (3 classes, 4 features)",
    Constants.DATASET_WINE: "Wine recognition (3 classes, 13 features)",
    Constants.DATASET_BREAST_CANCER: "Breast cancer Wisconsin diagnostic (2 classes, 30 features)",
    Constants.DATASET_DIGITS: "Optical recognition of handwritten digits (10 classes, 64 features)",
    Constants.DATASET_TITANIC: "Titanic passenger survival prediction (2 classes, 7 features)",
    "moons": "Make moons synthetic binary classification (2 classes, 2 features)",
    "blobs": "Make blobs synthetic multi-class classification (3 classes, 4 features)",
    "classification": "Make classification synthetic multi-class (3 classes, 10 features)",
}


def get_supported_datasets() -> list[str]:
    """Get list of supported dataset names.

    Returns:
        List of supported dataset name strings.
    """
    return list(SUPPORTED_DATASETS.keys())


def load_dataset(
    dataset: DatasetInput = Constants.DEFAULT_DATASET,
) -> tuple[ndarray, ndarray, list[str], list[str]]:
    """Load and prepare a dataset for decision tree classification.

    Supports standard scikit-learn toy datasets (iris, wine, breast_cancer, digits),
    synthetic generators (moons, blobs, classification), custom callables,
    Bunch objects, or existing data tuples.

    Args:
        dataset: Dataset name ('iris', 'wine', 'breast_cancer', 'digits', etc.),
            a callable returning a Bunch or (X, y), a Bunch object,
            or a tuple of (features, labels, [feature_names], [class_names]).

    Returns:
        Tuple of (features, labels, feature_names, class_names):
            - features: 2D numpy array of shape (n_samples, n_features).
            - labels: 1D numpy array of integer labels (0 to n_classes - 1).
            - feature_names: List of feature name strings.
            - class_names: List of class name strings.

    Raises:
        ValueError: If the dataset name is unknown or data format is invalid.
    """
    features: ndarray
    labels: ndarray
    feature_names: list[str] = []
    class_names: list[str] = []

    if isinstance(dataset, str):
        norm_name = dataset.strip().lower().replace("-", "_")
        if norm_name == Constants.DATASET_IRIS:
            bunch = load_iris()
            features = asarray(bunch.data, dtype=float)
            labels = asarray(bunch.target)
            feature_names = [str(f) for f in bunch.feature_names]
            class_names = [str(c) for c in bunch.target_names]
        elif norm_name == Constants.DATASET_WINE:
            bunch = load_wine()
            features = asarray(bunch.data, dtype=float)
            labels = asarray(bunch.target)
            feature_names = [str(f) for f in bunch.feature_names]
            class_names = [str(c) for c in bunch.target_names]
        elif norm_name in (Constants.DATASET_BREAST_CANCER, "cancer", "breast"):
            bunch = load_breast_cancer()
            features = asarray(bunch.data, dtype=float)
            labels = asarray(bunch.target)
            feature_names = [str(f) for f in bunch.feature_names]
            class_names = [str(c) for c in bunch.target_names]
        elif norm_name == Constants.DATASET_DIGITS:
            bunch = load_digits()
            features = asarray(bunch.data, dtype=float)
            labels = asarray(bunch.target)
            feature_names = [str(f) for f in bunch.feature_names]
            class_names = [str(c) for c in bunch.target_names]
        elif norm_name in ("moons", "make_moons"):
            moons_data = make_moons(n_samples=300, noise=0.2, random_state=Constants.RANDOM_STATE)
            features = asarray(moons_data[Constants.ZERO], dtype=float)
            labels = asarray(moons_data[Constants.ONE])
            feature_names = [f"feature_{i}" for i in range(features.shape[Constants.ONE])]
            class_names = ["class_0", "class_1"]
        elif norm_name in ("blobs", "make_blobs"):
            blobs_data = make_blobs(
                n_samples=300,
                centers=3,
                n_features=4,
                random_state=Constants.RANDOM_STATE,
                return_centers=False,
            )
            features = asarray(blobs_data[Constants.ZERO], dtype=float)
            labels = asarray(blobs_data[Constants.ONE])
            feature_names = [f"feature_{i}" for i in range(features.shape[Constants.ONE])]
            class_names = ["class_0", "class_1", "class_2"]
        elif norm_name in ("classification", "make_classification"):
            clf_data = make_classification(
                n_samples=300,
                n_features=10,
                n_informative=5,
                n_classes=3,
                random_state=Constants.RANDOM_STATE,
            )
            features = asarray(clf_data[Constants.ZERO], dtype=float)
            labels = asarray(clf_data[Constants.ONE])
            feature_names = [f"feature_{i}" for i in range(features.shape[Constants.ONE])]
            class_names = ["class_0", "class_1", "class_2"]
        elif norm_name in (Constants.DATASET_TITANIC, "openml_titanic"):
            from sklearn.datasets import fetch_openml

            openml_data = fetch_openml("titanic", version=1, as_frame=True)
            df = openml_data.frame
            feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
            x_df = df[feature_cols].copy()
            labels = asarray(df["survived"].astype(int).values)

            # Preprocess categorical and missing values
            x_df["sex"] = (x_df["sex"] == "female").astype(float)
            x_df["embarked"] = (
                x_df["embarked"].fillna("S").astype("category").cat.codes.astype(float)
            )
            x_df["age"] = x_df["age"].fillna(x_df["age"].median()).astype(float)
            x_df["fare"] = x_df["fare"].fillna(x_df["fare"].median()).astype(float)
            x_df["pclass"] = x_df["pclass"].astype(float)
            x_df["sibsp"] = x_df["sibsp"].astype(float)
            x_df["parch"] = x_df["parch"].astype(float)

            features = asarray(x_df.values, dtype=float)
            feature_names = feature_cols
            class_names = ["perished", "survived"]
        else:
            import sklearn.datasets

            loader: Callable[..., object] | None = getattr(
                sklearn.datasets, f"load_{norm_name}", None
            )
            if callable(loader):
                raw_bunch = loader()
                return load_dataset(raw_bunch)
            else:
                supported = ", ".join(get_supported_datasets())
                raise ValueError(f"Unknown dataset '{dataset}'. Supported datasets: {supported}")
    elif callable(dataset):
        raw_result = dataset()
        return load_dataset(raw_result)
    elif hasattr(dataset, "data") and hasattr(dataset, "target"):
        data_val = getattr(dataset, "data", None)
        target_val = getattr(dataset, "target", None)
        if data_val is None or target_val is None:
            raise ValueError("Dataset object must provide non-None 'data' and 'target' attributes")
        features = asarray(data_val, dtype=float)
        labels = asarray(target_val)
        raw_feature_names = getattr(dataset, "feature_names", None)
        feature_names = [str(f) for f in raw_feature_names] if raw_feature_names is not None else []
        target_names = getattr(dataset, "target_names", None)
        class_names = [str(c) for c in target_names] if target_names is not None else []
    elif isinstance(dataset, (tuple, list)):
        if len(dataset) == Constants.TWO:
            features = asarray(dataset[Constants.ZERO], dtype=float)
            labels = asarray(dataset[Constants.ONE])
        elif len(dataset) == Constants.FOUR:
            features = asarray(dataset[Constants.ZERO], dtype=float)
            labels = asarray(dataset[Constants.ONE])
            feature_names = [str(f) for f in dataset[Constants.TWO]]
            class_names = [str(c) for c in dataset[Constants.THREE]]
        else:
            raise ValueError(
                "Dataset tuple must have length 2 (features, labels) "
                "or 4 (features, labels, feature_names, class_names)"
            )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset).__name__}")

    # Ensure labels are 1D array
    labels = asarray(labels)
    unique_vals = unique(labels)

    # Check if labels are contiguous integers 0..K-1
    needs_encoding = False
    if not issubdtype(labels.dtype, integer):
        needs_encoding = True
    elif len(unique_vals) > Constants.ZERO and (
        unique_vals.min() != Constants.ZERO or unique_vals.max() != len(unique_vals) - Constants.ONE
    ):
        needs_encoding = True

    if needs_encoding:
        unique_classes, encoded_labels = unique(labels, return_inverse=True)
        labels = encoded_labels
        if not class_names or len(class_names) != len(unique_classes):
            class_names = [str(c) for c in unique_classes]

    # Validate or generate feature names
    n_features = features.shape[Constants.ONE] if features.ndim > Constants.ONE else Constants.ONE
    if not feature_names or len(feature_names) != n_features:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Validate or generate class names
    n_classes = len(unique(labels))
    if not class_names or len(class_names) != n_classes:
        class_names = [f"class_{i}" for i in range(n_classes)]

    return features, labels, feature_names, class_names
