"""
Module Name: tree_node.py
Description: Moduł zawierający klasę TreeNode, oraz funkcje potrzebne do generowania drzewa.
Authors: Adam Lipian, Mateusz Gawlik
Last Modified: 2025-01-25
Version: 1.0
"""

from collections import Counter
from pandas import DataFrame
from decision_tree.interval import Interval
from .split_methods import attribute_selection, split


class TreeNode:
    def __init__(self, interval: Interval):
        self.children: list[TreeNode] = []
        self.label: str = ""
        self.attribute: str = ""
        self.interval = interval

    def add_child(self, child) -> None:
        self.children.append(child)

    def merge(self, change: bool = False):
        """Łączy węzły, które mają tę samą etykietę"""
        for child in self.children:
            child.merge()
            for c in self.children:
                if c.label != "" and c.label == child.label and c != child:
                    self.children.remove(c)
                    child.interval += c.interval
                    change = True

        if len(self.children) == 1 and self.children[0].label != "":
            child = self.children[0]
            self.label = child.label
            self.attribute = ""
            self.children = []

        if change:
            self.merge()

    def classify(self, sample: DataFrame) -> str:
        if self.label != "":
            return self.label
        for child in self.children:
            if sample[self.attribute] in child.interval:
                return child.classify(sample)
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
        result += f" {self.attribute}"
        result += f" {self.interval}\n" if self.interval.intervals != [[0, 0]] else "\n"
        for i, leaf in enumerate(self.children):
            result += leaf.__str__(level + 1, is_last=(i == len(self.children) - 1), branches=branches)
        return result


def _same_label(samples: DataFrame) -> bool:
    """Sprawdza, czy wszystkie próbki mają tę samą etykietę"""
    return len(samples.iloc[:, -1].unique()) == 1


def _all_same(samples: DataFrame) -> bool:
    """Sprawdza, czy wszystkie przykłady posiadają identyczne wartości atrybutów"""
    return all(samples[col].nunique() == 1 for col in samples.columns)


def _most_common_label(samples: DataFrame) -> str:
    """Zwraca najczęstszą etykietę"""
    label = samples.iloc[:, -1]
    return Counter(label).most_common(1)[0][0]


def generate_tree(samples: DataFrame, attributes: list[str], split_method, num_splits: int = 2,
                  interval: Interval = Interval(0, 0), max_depth: int = float("inf"), _depth: int = 0) -> TreeNode:
    """Generuje drzewo decyzyjne (algorytm C5.0)"""
    node = TreeNode(interval)
    _depth += 1
    if _same_label(samples):
        node.label = samples.iloc[0, -1]
        return node

    if len(attributes) == 0 or _all_same(samples) or _depth == max_depth:
        node.label = _most_common_label(samples)
        return node

    best_attr = attribute_selection(samples, attributes)
    node.attribute = best_attr
    subsets = split(samples, best_attr, num_splits, split_method=split_method)
    new_attributes = [attr for attr in attributes if attr != best_attr]
    for subset, interval in subsets:
        subtree = generate_tree(subset, new_attributes, split_method, num_splits, interval, max_depth, _depth)
        node.add_child(subtree)

    return node
