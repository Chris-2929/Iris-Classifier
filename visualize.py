import numpy as np
import matplotlib.pyplot as plt


essential_colors = {
    "Iris-setosa": "red",
    "Iris-versicolor": "blue",
    "Iris-virginica": "green",
}


def scatter_petal(
    features: np.ndarray,
    labels: np.ndarray,
    means: dict[str, np.ndarray] | None = None,
    show: bool = False,
):
    """PetalLength vs PetalWidth colored by species; optionally overlays class means."""
    x = features[:, 2]  # PetalLengthCm
    y = features[:, 3]  # PetalWidthCm
    colors = [essential_colors[s] for s in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, c=colors, s=20, alpha=0.8)
    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.set_title("Iris: Petal scatter by species")

    if means is not None:
        for sp, mu in means.items():
            ax.scatter(
                mu[2],
                mu[3],
                s=200,
                marker="X",
                edgecolors="black",
                label=f"mean: {sp}",
                color=essential_colors[sp],
            )

    # Put legend outside the plot on the right
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax

def scatter_sepal(
    features: np.ndarray,
    labels: np.ndarray,
    means: dict[str, np.ndarray] | None = None,
    show: bool = False,
):
    """SepalLength vs SepalWidth colored by species; optionally overlays class means."""
    x = features[:, 0]  # SepalLengthCm
    y = features[:, 1]  # SepalWidthCm
    colors = [essential_colors[s] for s in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, c=colors, s=20, alpha=0.8)
    ax.set_xlabel("Sepal Length (cm)")
    ax.set_ylabel("Sepal Width (cm)")
    ax.set_title("Iris: Sepal scatter by species")

    if means is not None:
        for sp, mu in means.items():
            ax.scatter(
                mu[0],
                mu[1],
                s=200,
                marker="X",
                edgecolors="black",
                label=f"mean: {sp}",
                color=essential_colors[sp],
            )

    # Put legend outside the plot on the right
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax

def plot_per_class_accuracy(y_true, y_pred, classes: list[str], show: bool = False):
    """Bar chart showing per-class accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracies = []
    
    for class_name in classes:
        mask = (y_true == class_name)
        correct = np.sum(y_pred[mask] == y_true[mask])
        total = np.sum(mask)
        accuracies.append(correct / total if total > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(classes, accuracies, color="skyblue", edgecolor="black")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-class accuracy")

    for i, acc in enumerate(accuracies):
        # fix formatting
        if acc > 0.9:
            ax.text(i, acc - 0.05, f"{acc:.2f}", ha="center")
        else:
            ax.text(i, acc + 0.02, f"{acc:.2f}", ha="center")

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax

def plot_confusion_matrix(C: np.ndarray, classes: list[str], show: bool = False):
    """Plot confusion matrix as heatmap with annotations."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(C, interpolation="nearest", cmap="Blues")

    # Axis labels
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    # Annotate each cell
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            ax.text(
                j, i, str(C[i, j]),
                ha="center", va="center",
                color="white" if C[i, j] > C.max() / 2 else "black"
            )

    fig.colorbar(im, ax=ax)

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax