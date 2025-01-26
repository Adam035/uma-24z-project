"""
Module Name: decision_tree.py
Description: Główny plik uruchomieniowy.
Authors: Adam Lipian, Mateusz Gawlik
Last Modified: 2025-01-26
Version: 1.0
"""

from pandas import DataFrame
from decision_tree import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import click


def calculate_accuracy_and_precision(tree: TreeNode, samples: DataFrame, labels: list[str]) -> None:
    """Oblicza dokładność i wyświetla tablicę kontyngencji"""
    predicted = samples.apply(tree.classify, axis=1)
    conf_matrix = confusion_matrix(samples.iloc[:, -1], predicted, labels=labels)

    print(" ".join(labels))
    print(conf_matrix)

    correct = np.trace(conf_matrix)
    total = np.sum(conf_matrix)
    accuracy = round(100 * correct / total, 2)
    print(f"Accuracy: {accuracy}%")

    precisions = []
    for i, label in enumerate(labels):
        TP = conf_matrix[i, i]
        FP = sum(conf_matrix[:, i]) - TP
        if TP + FP > 0:
            precision = TP / (TP + FP)
        else:
            precision = 0.0
        precisions.append((label, round(precision, 2)))

    print("Precision:")
    for label, precision in precisions:
        print(f"{label}: {100 * precision}%")


@click.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--test-size', default=0.3)
@click.option('--random-state', default=42)
@click.option('--num-splits', default=2)
@click.option('--max-depth', default=1000)
@click.option('--split-method', default='equal')
def main(file_path, test_size, random_state, num_splits, max_depth, split_method):
    df, inputs = read_keel_file(file_path)
    labels = list(set(df.iloc[:, -1]))

    if split_method == 'entropy':
        method = entropy_split
    elif split_method == 'gini':
        method = gini_split
    else:
        method = equal_split

    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    tree = generate_tree(train, inputs, split_method=method, num_splits=num_splits, max_depth=max_depth)
    tree.merge()
    print(tree)

    calculate_accuracy_and_precision(tree, test, labels)
    print("(Test Set)")


if __name__ == "__main__":
    main()
