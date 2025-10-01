import numpy as np
import pytest

from data import load_iris_csv, split_deterministic
from model import compute_class_means, predict
from evaluate import accuracy, confusion_matrix


# ------------------------------
# Test 1: confusion matrix shape
# ------------------------------
def test_confusion_matrix_shape():
    # load dataset
    ids, X, y = load_iris_csv("data/Iris.csv")
    # split into train/test sets
    X_train, y_train, X_test, y_test = split_deterministic(X, y, train_per_class=35)
    # train simple nearest-mean model
    means = compute_class_means(X_train, y_train)
    # make predictions
    y_pred = predict(X_test, means)
    # compute confusion matrix
    C, classes = confusion_matrix(y_test, y_pred)

    # check that the shape is 3x3 (since there are 3 classes in the dataset)
    assert C.shape == (3, 3)
    # check that the set of classes matches expectation
    assert set(classes) == {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}


# ------------------------------
# Test 2: accuracy threshold
# ------------------------------
def test_accuracy_threshold():
    # load dataset
    ids, X, y = load_iris_csv("data/Iris.csv")
    X_train, y_train, X_test, y_test = split_deterministic(X, y, train_per_class=35)
    means = compute_class_means(X_train, y_train)
    y_pred = predict(X_test, means)

    acc = accuracy(y_test, y_pred)

    # for the dataset, nearest-mean classifier should be pretty strong.
    # enforce that accuracy must be at least 0.9.
    assert acc >= 0.9


# ------------------------------
# Test 3: trivial input case
# ------------------------------
def test_trivial_setosa_prediction():
    # load dataset
    ids, X, y = load_iris_csv("data/Iris.csv")
    X_train, y_train, X_test, y_test = split_deterministic(X, y, train_per_class=35)
    means = compute_class_means(X_train, y_train)

    # take a known "setosa" sample (first row)
    sample = X[0]

    # predict its label
    pred = predict(np.array([sample]), means)[0]

    # it should come out as "Iris-setosa"
    assert pred == "Iris-setosa"
