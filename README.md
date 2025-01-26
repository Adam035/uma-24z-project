# Decision Tree with Split Method Selection

## Installation

1. Install all required libraries using the `requirements.txt` file:

    ```bash
    pip3 install -r requirements.txt
    ```

2. Make sure you have all the necessary dependencies installed. You can use a virtual environment to avoid dependency conflicts.

## Usage

To run the program, use the following command in your terminal:

```bash
python3 main.py <file_path> [--test-size TEST_SIZE] [--random-state RANDOM_STATE] [--num-splits NUM_SPLITS] [--max-depth MAX_DEPTH] [--split-method SPLIT_METHOD]
```

### Arguments

* `file_path` (required): The path to the input KEEL file (with data for analysis).

### Options

* `--test-size`: The proportion of the data to be used as the test set (default is 0.3).
* `--random-state`: The random seed value (default is 42).
* `--num-splits`: The number of splits used for data division (default is 2).
* `--max-depth`: The maximum depth of the decision tree (default is 1000).
* `--split-method`: The splitting method. Possible values:
    * `equal`: Equal split (default).
    * `entropy`: Split based on entropy.
    * `gini`: Split based on Gini index.

### Example Command

To run the program on a KEEL file with default settings:

```bash
python3 decision_tree.py data-sets/magic.dat --test-size 0.3 --random-state 42 --num-splits 3 --max-depth 4
```
