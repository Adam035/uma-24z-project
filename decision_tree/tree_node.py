from collections import Counter
from pandas import DataFrame

class TreeNode:
    def __init__(self, min_value: float, max_value: float):
        self.leaves: list[TreeNode] = []
        self.label: str = ""
        self.attribute: str = ""
        self.min_value: float = min_value
        self.max_value: float = max_value

    def add_leaf(self, leaf) -> None:
        self.leaves.append(leaf)

    def classify(self, sample: DataFrame) -> str:
        if self.label != "":
            return self.label
        for leaf in self.leaves:
            if leaf.min_value <= sample[self.attribute] < leaf.max_value:
                label = leaf.classify(sample)
                if label:
                    return label
                break

def same_label(samples: DataFrame) -> bool:
    return len(samples.iloc[:, -1].unique()) == 1

def all_same(samples: DataFrame) -> bool:
    return all(samples[col].nunique() == 1 for col in samples.columns)

def most_common_label(samples: DataFrame) -> str:
    label = samples.iloc[:, -1]
    return Counter(label).most_common(1)[0][0]

def attribute_selection(attributes: list[str]) -> str:
    return attributes[0]

def generate_tree(samples: DataFrame, attributes: list[str], split_method, min_value: float = 0, max_value: float = 0) -> TreeNode:
    node = TreeNode(min_value, max_value)

    if same_label(samples):
        node.label = samples.iloc[0, -1]
        return node

    if len(attributes) == 0 or all_same(samples):
        node.label = most_common_label(samples)
        return node

    best_attr = attribute_selection(attributes)
    node.attribute = best_attr
    subsets = split_method(samples, best_attr)

    new_attributes = [attr for attr in attributes if attr != best_attr]
    for subset, range_values in subsets:
        node.add_leaf(generate_tree(subset, new_attributes, split_method, range_values[0], range_values[1]))

    return node
