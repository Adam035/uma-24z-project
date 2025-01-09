from pandas import DataFrame
from typing import Tuple
import math

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


def entropy(samples:DataFrame) -> float:
    classes = samples.iloc[:,-1].value_counts()
    samples_count = samples.shape[0]
    entropy = 0
    for _, class_count in classes.items():
        p = class_count/samples_count 
        entropy -= p*math.log(p,2)
    return entropy
        
def gini(samples:DataFrame) -> float:
    classes = samples.iloc[:,-1].value_counts()
    samples_count = samples.shape[0]
    gini = 1
    for _, class_count in classes.items():
        p = class_count/samples_count 
        gini -= p*p
    return gini

def attribute_selection(samples: DataFrame, split_func, index_func=entropy):
    attributes = [i for i in samples.columns][:-1]
    
    whole_set_count = samples.shape[0]
    whole_set_index = index_func(samples)
    
    information_gain: dict = {}
    for attr in attributes:
        subsets = split_func(samples, attr, num_splits=4)
        index_after_division = 0
        for subset in subsets:
            samples_count = subset[0].shape[0]
            subset_index = index_func(subset[0])
            index_after_division += samples_count/whole_set_count * subset_index
        inf_gain_attr = whole_set_index - index_after_division
        information_gain[attr] = inf_gain_attr
    max_inf_gain = max(information_gain, key=information_gain.get)
    return max_inf_gain

    
    

if __name__ == '__main__':
    samples = DataFrame({"A1": ["CZer","Ziel","Czer","Czer","Ziel","Ziel"], "A2": ["Mały", "Średni", "Mały", "Duzy","Średni","Duży"],"A3": ["Gładki", "Gładki", "Chropowaty", "Gładki", 'Chropowaty', "Gładki"] , "Class": [1,0,1,1,0,0]})
    samples_cont = DataFrame({
        "Age": [20, 30, 35, 40, 45, 60],
        "Income": [30, 40, 60, 60, 70, 80],
        "Class": ["Tak", "Tak", "Nie", "Tak", "Nie", "Nie"]
    })
    #print(gini(samples))
    print(attribute_selection(samples_cont, quantile_split, gini))