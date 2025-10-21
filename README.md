# Iris Flower Classification

This project trains and evaluates simple classifiers on an Iris dataset.

Project includes:
- Loading and exploring data
- Splitting train/test sets
- Training a nearest-mean classifier and KNN baseline
- Evaluating with accuracy and confusion matrices
- Visualizing results
    * Confusion matrix heatmap
    * Per-class accuracy bar chart
    * Sepal scatterplot
    * Petal scatterplot

## Dataset

This project uses the [Iris dataset from Kaggle](https://www.kaggle.com/datasets/uciml/iris).  
The CSV (`data/Iris.csv`) is included in this repository for convenience.

- License: **CC0: Public Domain**  
- Source: [Kaggle - Iris Dataset](https://www.kaggle.com/datasets/uciml/iris)

## Quickstart

Clone the repo and install requirements:

git clone https://github.com/Chris-2929/iris-classifier.git
cd iris-classifier
pip install -r requirements.txt

## Run the main program:

py main.py --path=data/Iris.csv --train-size=30 --plot

Example output:

[INFO] Loading dataset from Iris.csv
[INFO] Loaded 150 samples with 4 features
[INFO] Nearest-mean (raw) accuracy: 0.983
[INFO] KNN (raw) accuracy: 0.983
[INFO] Nearest-mean (scaled) accuracy: 0.883
[INFO] KNN (scaled) accuracy: 0.950


## Visualizations

The visualize.py module plots:

Confusion matrices

Accuracy comparisons

Scatterplot of SepalLengthCm, SepalWidthCm

Scatterplot of PetalLengthCm, PetalWidthCm

## Running Tests

This repo includes a few basic tests with pytest.

pytest

## Project Structure

```plaintext
├── analysis.py       # Training and experiment logic
├── data.py           # Data loading and splitting
├── evaluate.py       # Accuracy + confusion matrix
├── main.py           # CLI entry point
├── model.py          # Nearest-mean classifier
├── visualize.py      # Plotting helpers
├── tests_basic.py    # Basic unit tests
├── requirements.txt  # Dependencies
└── data/
    └── Iris.csv      # Dataset (CC0)
```

## License

Code: MIT License (see LICENSE)

Dataset: Iris dataset (CC0: Public Domain)

## Development Note

Used ChatGPT-5 as a coding assistant for learning purposes
but all code was developed, reviewed, and tested by me
