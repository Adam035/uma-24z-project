from decision_tree import read_keel_file, generate_tree, SplitMethod

file_path = "data-sets/magic.dat"
df, inputs, outputs = read_keel_file(file_path)

tree = generate_tree(df, inputs, SplitMethod.MEDIAN)

print(tree)
