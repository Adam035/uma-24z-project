"""
Module Name: read_data.py
Description: Moduł do wczytywania danych z pliku DAT.
Authors: Adam Lipian, Mateusz Gawlik
Last Modified: 2025-01-25
Version: 1.0
"""

import re
import pandas as pd


def read_keel_file(file_path):
    """Odczyt pliku DAT"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data_start = None
    inputs = []
    columns = []
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('@attribute'):
            columns.append(line.split()[1])
        if line.startswith('@inputs'):
            inputs = re.split(r',\s*|\s+', line)[1:]
        if line.startswith('@data'):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError("'@data' not found")

    data_lines = lines[data_start:]
    data = [line.strip().split(',') for line in data_lines if line.strip()]
    df = pd.DataFrame(data, columns=columns)
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')
    return df, inputs
