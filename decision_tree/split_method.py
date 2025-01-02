from pandas import DataFrame
from typing import Tuple

def median_split(samples: DataFrame, attribute: str, num_splits: int = 2) -> list[Tuple[DataFrame, list[float]]]:
    median = samples[attribute].median()
    left_subset = samples[samples[attribute] <= median], [float("-inf"), median]
    right_subset = samples[samples[attribute] > median], [median, float("inf")]
    return [left_subset, right_subset]


def quantile_split(samples: DataFrame, attribute: str, num_splits: int = 4) -> list[Tuple[DataFrame, list[float]]]:
    quantiles = [i / num_splits for i in range(num_splits + 1)]
    thresholds = samples[attribute].quantile(q=quantiles).values.tolist()
    thresholds[0], thresholds[-1] = float("-inf"), float("inf")
    subsets = []
    for i in range(num_splits):
        subset = samples[(thresholds[i] <= samples[attribute]) & (samples[attribute] < thresholds[i+1])]
        subsets.append((subset, [thresholds[i], thresholds[i+1]]))
    return subsets
