import numpy as np
from typing import Dict


def compute_class_means(X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Mean feature vector for each class label.

    Args:
        X: 
        y: Species labels
    Returns:
        Mean vectors for each species in a dictionary
    """
    # initialize means as empty dictionary
    means: Dict[str, np.ndarray] = {}
    for sp in np.unique(y):
        # averaging each column
        means[sp] = X[y == sp].mean(axis=0) 
    # return the dictionary with species : mean vector as the key : value pair
    return means


def nearest_mean_classifier(x: np.ndarray, means: Dict[str, np.ndarray]) -> str:
    """
    Predict label for a single sample by nearest squared Euclidean distance to class means.
    
    Returns:
        Best species prediction as string
    """
    x = np.asarray(x, dtype=float)
    best_sp = None
    best_d2 = np.inf
    for sp, mu in means.items():
        d2 = np.sum((x - mu) ** 2)
        if d2 < best_d2:
            best_sp = sp
            best_d2 = d2
    return best_sp


def predict(X: np.ndarray, means: Dict[str, np.ndarray]) -> np.ndarray:
    """Vectorized prediction for a 2D array of samples."""
    return np.array([nearest_mean_classifier(row, means) for row in X])