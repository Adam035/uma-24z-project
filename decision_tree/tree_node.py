from collections import Counter
from pandas import DataFrame

from decision_tree.interval import Interval
from .split_method import attribute_selection


class TreeNode:
    def __init__(self, interval: Interval):
        self.leaves: list[TreeNode] = []
        self.label: str = ""
        self.attribute: str = ""
        self.interval = interval

    def add_leaf(self, leaf) -> None:
        self.leaves.append(leaf)

    def merge(self, change: bool = False):
        for leaf in self.leaves:
            leaf.merge()
            for l in self.leaves:
                if l.label != "" and l.label == leaf.label and l != leaf:
                    self.leaves.remove(l)
                    leaf.interval += l.interval
                    change = True

        if len(self.leaves) == 1 and self.leaves[0].label != "":
            single_leaf = self.leaves[0]
            self.label = single_leaf.label
            self.attribute = ""
            self.leaves = []

        if change:
            self.merge()


    def classify(self, sample: DataFrame) -> str:
        if self.label != "":
            return self.label
        for leaf in self.leaves:
            if sample[self.attribute] in leaf.interval:
                label = leaf.classify(sample)
                if label:
                    return label
                break
        return ""

    def __str__(self, level=0, is_last=True, branches=None) -> str:
        if branches is None:
            branches = [False]
        result = ""
        for branch in branches:
            result += "│  " if branch else "   "
        result += "└" if is_last else "├"
        if result[-1] == "└":
            branches.append(False)
        elif result[-1] == "├":
            branches.append(True)
        result += "──"
        if self.label != "":
            result += f" {self.label}"
            branches.pop()
            if is_last:
                while branches and not branches[-1]:
                    branches.pop()
                if branches and branches[-1]:
                    branches.pop()
        else:
            result += "┐"
        result += f" {self.attribute} {self.interval}\n"
        for i, leaf in enumerate(self.leaves):
            result += leaf.__str__(level + 1, is_last=(i == len(self.leaves) - 1), branches=branches)
        return result


def same_label(samples: DataFrame) -> bool:
    return len(samples.iloc[:, -1].unique()) == 1

def all_same(samples: DataFrame) -> bool:
    return all(samples[col].nunique() == 1 for col in samples.columns)

def most_common_label(samples: DataFrame) -> str:
    label = samples.iloc[:, -1]
    return Counter(label).most_common(1)[0][0]

def generate_tree(samples: DataFrame, attributes: list[str], split_method, num_splits: int = 2, interval: Interval = Interval(0, 0), max_depth: int = float("inf"), _depth: int = 0) -> TreeNode:
    node = TreeNode(interval)
    _depth += 1
    if same_label(samples):
        node.label = samples.iloc[0, -1]
        return node

    if len(attributes) == 0 or all_same(samples) or _depth == max_depth:
        node.label = most_common_label(samples)
        return node

    best_attr = attribute_selection(samples, attributes)
    node.attribute = best_attr
    subsets = split_method(samples, best_attr, num_splits)

    new_attributes = [attr for attr in attributes if attr != best_attr]
    for subset, interval in subsets:
        subtree = generate_tree(subset, new_attributes, split_method, num_splits, interval, max_depth, _depth)
        node.add_leaf(subtree)

    return node
