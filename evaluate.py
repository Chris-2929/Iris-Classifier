import numpy as np


def accuracy(y_true, y_pred) -> float:
    """Overall fraction correct."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred) -> tuple[np.ndarray, list[str]]:
    """
    3x3 confusion counts with rows=true and cols=pred; 
    
    Returns:
        (C, classes order).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = list(np.unique(y_true))
    idx = {c: i for i, c in enumerate(classes)}
    # idx = {np.str_('Iris-setosa'): 0, np.str_('Iris-versicolor'): 1, np.str_('Iris-virginica'): 2}
    C = np.zeros((len(classes), len(classes)), dtype=int)
    # [[0 0 0]
    #  [0 0 0]
    #  [0 0 0]]
    for t, p in zip(y_true, y_pred):
        C[idx[t], idx[p]] += 1
    return C, classes
