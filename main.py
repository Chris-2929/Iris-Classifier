from data import load_iris_csv, split_deterministic
from model import compute_class_means, predict
from evaluate import accuracy, confusion_matrix
from visualize import plot_confusion_matrix, plot_per_class_accuracy, scatter_petal, scatter_sepal
from analysis import baseline_knn
from analysis import scaled_experiment

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def parse_args():
    parser = argparse.ArgumentParser(description="Nearest-mean Iris classifier")

    parser.add_argument("--path",default="Iris.csv",help="Path to Iris dataset CSV (default: Iris.csv)")
    parser.add_argument("--train-size",type=int,default=35,help="Number of training samples per class (default: 35)")
    parser.add_argument("--plot",action="store_true",help="If set, show scatter/plots")

    return parser.parse_args()

def main(path: str, train_size: int, plot: bool):
    
    # Handling errors -----
    try:
        logging.info("Loading dataset from Iris.csv")
        ids, features, labels = load_iris_csv(path)
        logging.info("Loaded 150 samples with 4 features")
    except (FileNotFoundError, ValueError) as e:
        logging.error("Error loading dataset: %s", e)
        return
    
    if not (1 <= train_size < 50):
        logging.error("train-size must be between 1 and 49 (got %s)", train_size)
        return
    # ---------------------

    X_train, y_train, X_test, y_test = split_deterministic(features, labels, train_per_class=train_size)

    means_train = compute_class_means(X_train, y_train)
    y_pred = predict(X_test, means_train)

    acc = accuracy(y_test, y_pred)
    logging.info(f"Nearest-mean (raw) accuracy: {acc:.3f}")

    C, classes = confusion_matrix(y_test, y_pred)
    # logging.info("Confusion matrix:\n%s", C) # prints to terminal

    baseline_acc = baseline_knn(X_train, y_train, X_test, y_test, k=5)
    logging.info(f"KNN (raw) accuracy: {baseline_acc:.3f}")

    results = scaled_experiment(X_train, y_train, X_test, y_test, k=5)
    for name, acc in results.items():
        logging.info("%s accuracy: %.3f", name, acc)
    
    if plot == True:
        scatter_petal(features, labels, means=means_train, show=True)
        scatter_sepal(features, labels, means=means_train, show=True)
        plot_confusion_matrix(C, classes, show=True)
        plot_per_class_accuracy(y_test, y_pred, classes, show=True)

if __name__ == "__main__":
    args = parse_args()
    main(args.path, args.train_size, args.plot)