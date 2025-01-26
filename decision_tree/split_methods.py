"""
Module Name: split_methods.py
Description: Moduł zawierający funkcje do wybooru miejsca przedziału oraz najlepszego atrybutu.
Authors: Adam Lipian, Mateusz Gawlik
Last Modified: 2025-01-25
Version: 1.0
"""

import numpy as np
from pandas import DataFrame
import pandas as pd
from typing import Tuple
import math
from decision_tree.interval import Interval


def equal_split(samples: pd.DataFrame, attribute: str, num_splits: int = 4) -> list[float]:
    """Dzieli wartości atrybutu na równe przedziały według kwantyli."""
    quantiles = [i / num_splits for i in range(num_splits + 1)]
    thresholds = samples[attribute].quantile(q=quantiles).values.tolist()
    return thresholds


def _entropy(samples: DataFrame) -> float:
    """Oblicza entropię"""
    if samples.empty:
        return 0
    target_column = samples.columns[-1]
    labels = samples[target_column].value_counts()
    samples_count = samples.shape[0]
    entropy_value = 0
    for _, class_count in labels.items():
        p = class_count / samples_count
        entropy_value -= p * math.log(p, 2)
    return entropy_value


def _conditional_entropy(samples, attribute, threshold):
    """Oblicza warunkową entropię"""
    if samples.empty:
        return 0

    left_group = samples[samples[attribute] <= threshold]
    right_group = samples[samples[attribute] > threshold]
    total_size = len(samples)
    if total_size == 0:
        return 0

    left_entropy = _entropy(left_group) if not left_group.empty else 0.0
    right_entropy = _entropy(right_group) if not right_group.empty else 0.0
    cond_entropy = (len(left_group) / total_size) * left_entropy + (len(right_group) / total_size) * right_entropy
    return cond_entropy


def entropy_split(samples, attribute, num_splits):
    """Wybiera najlepsze progi, ze względu na entropię"""
    min_value, max_value = samples[attribute].min(), samples[attribute].max()
    interval_size = (max_value - min_value) / (num_splits + 1)
    thresholds = [
        np.mean([min_value + i * interval_size, min_value + (i + 1) * interval_size])
        for i in range(num_splits + 1)
    ]
    entropy_list = [
        (threshold, _conditional_entropy(samples, attribute, threshold))
        for threshold in thresholds
    ]
    sorted_thresholds = sorted(entropy_list, key=lambda x: x[1])
    best_thresholds = [threshold for threshold, _ in sorted_thresholds[:num_splits + 1]]
    return sorted(best_thresholds)


def _gini(samples: DataFrame) -> float:
    """Oblicza gini index"""
    if samples.empty:
        return 0
    labels = samples.iloc[:, -1].value_counts()
    samples_count = samples.shape[0]
    gini_value = 1
    for _, class_count in labels.items():
        p = class_count / samples_count
        gini_value -= p * p
    return gini_value


def _conditional_gini(samples, attribute, threshold):
    """Oblicza warunkowy gini index"""
    if samples.empty:
        return 0

    left_group = samples[samples[attribute] <= threshold]
    right_group = samples[samples[attribute] > threshold]
    total_size = len(samples)
    if total_size == 0:
        return 0

    left_gini = _gini(left_group) if not left_group.empty else 0
    right_gini = _gini(right_group) if not right_group.empty else 0
    cond_gini = (len(left_group) / total_size) * left_gini + (len(right_group) / total_size) * right_gini
    return cond_gini


def gini_split(samples, attribute, num_splits):
    """Wybiera najlepsze progi, ze względu na entropię"""
    min_value, max_value = samples[attribute].min(), samples[attribute].max()
    interval_size = (max_value - min_value) / (num_splits + 1)
    thresholds = [
        np.mean([min_value + i * interval_size, min_value + (i + 1) * interval_size])
        for i in range(num_splits + 1)
    ]
    gini_list = [
        (threshold, _conditional_gini(samples, attribute, threshold))
        for threshold in thresholds
    ]
    sorted_thresholds = sorted(gini_list, key=lambda x: x[1])
    best_thresholds = [threshold for threshold, _ in sorted_thresholds[:num_splits + 1]]
    return sorted(best_thresholds)


def split(samples: DataFrame, attribute: str, num_splits: int = 4,
          split_method = equal_split) -> list[Tuple[DataFrame, Interval]]:
    """Podział na podzbiory"""
    thresholds = split_method(samples, attribute, num_splits)
    thresholds[0], thresholds[-1] = float("-inf"), float("inf")
    subsets = []
    for i in range(num_splits):
        subset = samples[(thresholds[i] <= samples[attribute]) & (samples[attribute] < thresholds[i + 1])]
        interval = Interval(thresholds[i], thresholds[i + 1])
        if not subset.empty:
            subsets.append((subset, interval))
        elif len(subsets) != 0:
            temp_subset, temp_interval = subsets[-1]
            temp_interval += interval
            subsets.pop()
            subsets.append((temp_subset, temp_interval))
    return subsets


def attribute_selection(samples: DataFrame, attributes: list[str], index_func=_entropy):
    """Wybór najlepszego atrybutu"""
    whole_set_count = samples.shape[0]
    whole_set_index = index_func(samples)

    information_gain: dict = {}
    for attr in attributes:
        subsets = split(samples, attr, num_splits=10)
        index_after_division = 0
        for subset, _ in subsets:
            samples_count = subset.shape[0]
            if samples_count == 0:
                continue
            subset_index = index_func(subset)
            index_after_division += samples_count / whole_set_count * subset_index
        inf_gain_attr = whole_set_index - index_after_division
        information_gain[attr] = inf_gain_attr
    max_inf_gain = max(information_gain, key=information_gain.get)
    return max_inf_gain


