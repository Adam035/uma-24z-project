from decision_tree import read_keel_file, generate_tree, median_split
from sklearn.metrics import confusion_matrix
import numpy as np

file_path = "data-sets/magic.dat"

if __name__ == "__main__":
    df, inputs = read_keel_file(file_path)
    labels = list(set(df.iloc[:, -1]))

    tree = generate_tree(df, inputs, median_split)
    predicted = df.apply(tree.classify, axis=1)

    conf_matrix = confusion_matrix(df.iloc[:, -1], predicted, labels=labels)

    print(" ".join(labels))
    print(conf_matrix)

    correct = np.trace(conf_matrix)
    total = np.sum(conf_matrix)
    accuracy = 100 * correct / total

    print(f"Accuracy: {accuracy}%")