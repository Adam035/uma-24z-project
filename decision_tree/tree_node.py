import statistics

from collections import Counter
from pandas import DataFrame
from decision_tree.split_method import SplitMethod

class TreeNode:
    def __init__(self):
        self.leaves = []
        self.label = None
        self.attribute = None
        self.threshold = 0

    def add_leaf(self, leaf):
        self.leaves.append(leaf)

    def __str__(self, depth=0):
        if self.label is not None:
            return "  " * depth + f"Label: {self.label}"
        result = "  " * depth + f"Attribute: {self.attribute}, Threshold: {self.threshold}\n"
        for i, leaf in enumerate(self.leaves):
            result += "  " * (depth + 1) + f"Leaf {i + 1}:\n"
            result += leaf.__str__(depth + 2) + "\n"

        return result

def same_label(samples: DataFrame) -> bool:
    return len(samples.iloc[:, -1].unique()) == 1

def all_unique(samples: DataFrame) -> bool:
    return all(samples[col].nunique() == 1 for col in samples.columns)

def most_common_label(samples: DataFrame) -> str:
    label = samples.iloc[:, -1]
    return Counter(label).most_common(1)[0][0]

def attribute_selection(attributes: [str]) -> str:
    return attributes[0]

def median_split(node: TreeNode, samples: DataFrame, attribute: str) -> [DataFrame]:
    median = statistics.median(samples[attribute])
    node.threshold = median
    left_subset = samples[samples[attribute] <= median]
    right_subset = samples[samples[attribute] > median]
    return [left_subset, right_subset]

def splitting_criterion(split_method: SplitMethod, node: TreeNode, samples: DataFrame, attribute: str) -> [DataFrame]:
    if split_method == SplitMethod.MEDIAN:
        return median_split(node, samples, attribute)

def generate_tree(samples: DataFrame, attributes: [str], split_method: SplitMethod) -> TreeNode:
    node = TreeNode()

    if same_label(samples):
        node.label = samples.iloc[0, -1]
        return node

    if len(attributes) == 0 or all_unique(samples):
        node.label = most_common_label(samples)
        return node

    best_attr = attribute_selection(attributes)
    node.attribute = best_attr
    subsets = splitting_criterion(split_method, node, samples, best_attr)

    new_attributes = [attr for attr in attributes if attr != best_attr]
    for subset in subsets:
        node.add_leaf(generate_tree(subset, new_attributes, split_method))

    return node
