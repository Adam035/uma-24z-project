from decision_tree import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


def calculate_accuracy(samples, labels):
    predicted = samples.apply(tree.classify, axis=1)
    conf_matrix = confusion_matrix(samples.iloc[:, -1], predicted, labels=labels)

    print(" ".join(labels))
    print(conf_matrix)

    correct = np.trace(conf_matrix)
    total = np.sum(conf_matrix)
    accuracy = 100 * correct / total

    print(f"Accuracy: {accuracy}%", end=" ")

file_path = "data-sets/magic.dat"

if __name__ == "__main__":
    df, inputs = read_keel_file(file_path)
    labels = list(set(df.iloc[:, -1]))

    train, test = train_test_split(df, test_size=0.3, random_state=42)

    tree = generate_tree(train, inputs, quantile_split, num_splits=5, max_depth=5)
    print(tree)

    calculate_accuracy(train, labels)
    print("(Train Set)", end="\n\n")
    calculate_accuracy(test, labels)
    print("(Test Set)")
