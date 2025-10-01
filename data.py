import numpy as np
import os

def load_iris_csv(path: str = "data/Iris.csv") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Iris CSV and return (ids, features[*,4], labels[str])."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
    
    if data.shape[1] < 6:
        raise ValueError("CSV missing expected columns (need ID, 4 features, label)")
    
    ids = data[:, 0].astype(int)
    X = data[:, 1:5].astype(float)
    y = data[:, 5]
    return ids, X, y


def split_deterministic(
    features: np.ndarray,
    labels: np.ndarray,
    train_per_class: int = 35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic split matching Iris ordering (50 per class).

    Returns X_train, y_train, X_test, y_test.
    """

    # training data
    setosa_train = features[0:train_per_class]
    versicolor_train = features[50 : 50 + train_per_class]
    virginica_train = features[100 : 100 + train_per_class]
    X_train = np.vstack([setosa_train, versicolor_train, virginica_train])

    # training data labels
    y_train = np.array(
        ["Iris-setosa"] * train_per_class
        + ["Iris-versicolor"] * train_per_class
        + ["Iris-virginica"] * train_per_class
    )

    # testing data
    setosa_test = features[train_per_class:50]
    versicolor_test = features[50 + train_per_class : 100]
    virginica_test = features[100 + train_per_class : 150]
    X_test = np.vstack([setosa_test, versicolor_test, virginica_test])

    # testing data labels (answer key)
    test_count = 50 - train_per_class
    y_test = np.array(
        ["Iris-setosa"] * test_count
        + ["Iris-versicolor"] * test_count
        + ["Iris-virginica"] * test_count
    )

    return X_train, y_train, X_test, y_test