from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from model import compute_class_means, predict
from evaluate import accuracy

def baseline_knn(X_train, y_train, X_test, y_test, k: int = 5) -> float:
    """
    Classify test samples using a simple k-Nearest Neighbors approach.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        k: Number of neighbors

    Returns:
        Accuracy of KNN on the test set.
    """
    clf = KNeighborsClassifier(n_neighbors=k)
    # train on the training set
    clf.fit(X_train, y_train)
    # predict on the test set
    y_pred = clf.predict(X_test)         
    return accuracy_score(y_test, y_pred)


def scaled_experiment(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    k: int = 5
) -> dict[str, float]:
    """
    Evaluate model performance on standardized (scaled) features.

    Returns a dictionary with two entries:
      - Nearest-mean (scaled)
      - KNN (scaled)
    """

    results: dict[str, float] = {}

    # -----------------
    # Scaled features
    # -----------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    means_scaled = compute_class_means(X_train_scaled, y_train)
    y_pred_scaled = predict(X_test_scaled, means_scaled)
    
    results["Nearest-mean (scaled)"] = accuracy(y_test, y_pred_scaled)
    results["KNN (scaled)"] = baseline_knn(X_train_scaled, y_train, X_test_scaled, y_test, k=k)

    return results
