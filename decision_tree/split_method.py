from pandas import DataFrame

def median_split(samples: DataFrame, attribute: str) -> list[DataFrame]:
    median = samples[attribute].median()
    left_subset = samples[samples[attribute] <= median], [-float("inf"), median]
    right_subset = samples[samples[attribute] > median], [median, float("inf")]
    return [left_subset, right_subset]