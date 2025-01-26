"""
Module Name: interval.py
Description: Klasa reprezentująca przedział liczbowy.
Authors: Adam Lipian, Mateusz Gawlik
Last Modified: 2025-01-25
Version: 1.0
"""


class Interval:
    def __init__(self, min_value: float, max_value: float):
        self.intervals = [[min_value, max_value]]

    def __str__(self):
        result = ""
        for interval in self.intervals:
            result += f"({interval[0]}, {interval[1]}) "
        return result.strip()

    def __add__(self, other):
        new_intervals = self.intervals + other.intervals
        new_intervals.sort(key=lambda x: (x[0], x[1]))
        merged_intervals = []
        current_start, current_end = new_intervals[0]

        for start, end in new_intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_intervals.append([current_start, current_end])
                current_start, current_end = start, end

        merged_intervals.append([current_start, current_end])
        result = Interval(0, 0)
        result.intervals = merged_intervals
        return result

    def __contains__(self, value):
        for start, end in self.intervals:
            if start <= value <= end:
                return True
        return False

