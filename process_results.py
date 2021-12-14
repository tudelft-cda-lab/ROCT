import matplotlib.pyplot as plt

from roct.upper_bound import maximum_adversarial_accuracy

import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=0.8)

import pandas as pd

import numpy as np

# Avoid type 3 fonts
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from tqdm import tqdm

import os

import json

result_dir = "out/results/"
figure_dir = "out/figures/"
data_dir = "data/"

results = []
for result_name in tqdm(os.listdir(result_dir)):
    filename = result_dir + result_name

    with open(filename) as file:
        result = json.load(file)

    dataset, algorithm, epsilon = result_name[:-5].split("_")

    # Load datasets
    X_train = np.load(data_dir + f"X_train_{dataset}.npy")
    X_test = np.load(data_dir + f"X_test_{dataset}.npy")
    y_train = np.load(data_dir + f"y_train_{dataset}.npy")
    y_test = np.load(data_dir + f"y_test_{dataset}.npy")

    if algorithm == "treant" or "rc2-maxsat":
        # Count a timeout if the algorithm selected a tree with depth 0
        timeout = "leaf" in result["model"][0] and result["best_depth"] != 0
    else:
        timeout = False

    # Determine adversarial accuracy bound
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    Delta_l = Delta_r = np.full(X.shape[1], fill_value=float(epsilon))
    adv_acc_bound = maximum_adversarial_accuracy(X, y, Delta_l, Delta_r)
    train_adv_acc_bound = maximum_adversarial_accuracy(X_train, y_train, Delta_l, Delta_r)
    test_adv_acc_bound = maximum_adversarial_accuracy(X_test, y_test, Delta_l, Delta_r)

    results.append(
        (
            dataset,
            epsilon,
            algorithm,
            result["best_depth"],
            result["train_accuracy"],
            result["train_adv_accuracy"],
            result["test_accuracy"],
            result["test_adv_accuracy"],
            adv_acc_bound,
            train_adv_acc_bound,
            test_adv_acc_bound,
            timeout,
        )
    )

columns = [
    "Dataset",
    "Epsilon",
    "Algorithm",
    "Best depth",
    "Train accuracy",
    "Train adversarial accuracy",
    "Test accuracy",
    "Test adversarial accuracy",
    "Adversarial accuracy bound",
    "Train adversarial accuracy bound",
    "Test adversarial accuracy bound",
    "Timeout",
]
result_df = pd.DataFrame(results, columns=columns)

algorithm_names = {
    "tree": "Decision Tree",
    "treant": "TREANT",
    "groot": "GROOT",
    "lsu-maxsat": "LSU-MaxSAT",
    "rc2-maxsat": "RC2-MaxSAT",
    "milp": "MILP",
    "bin-milp": "Binary-MILP",
    "milp-warm": "MILP-warm",
    "bin-milp-warm": "Binary-MILP-warm",
}
result_df["Algorithm"] = result_df["Algorithm"].map(algorithm_names)
print(result_df["Algorithm"].value_counts())

mean_scores = result_df[["Algorithm", "Test adversarial accuracy"]].groupby("Algorithm").mean()
order = mean_scores.sort_values(by="Test adversarial accuracy").index

result_table = result_df.pivot_table(
    values="Test adversarial accuracy",
    index=["Dataset", "Epsilon"],
    columns="Algorithm",
    fill_value=0.0,
)
result_table = result_table[list(algorithm_names.values())]
result_table = result_table[order]

latex_result_table = result_table.copy()

# Output latex table with bold values
format_string = "%.3f"
maxima = latex_result_table.max(axis=1)
for i, row in latex_result_table.iterrows():
    latex_result_table.loc[row.name] = row.apply(
        lambda x: ("\\textbf{%s}" % format_string % x)
        if x == maxima[i]
        else ("%s" % format_string % x)
    )
print(latex_result_table.to_latex(escape=False))
latex_result_table.to_latex("out/figures/result_table.tex", escape=False)

# Output table of selected max_depth values
depth_table = result_df.pivot_table(
    values="Best depth",
    index=["Dataset", "Epsilon"],
    columns="Algorithm",
    fill_value=0.0,
)
depth_table = depth_table[list(algorithm_names.values())]
depth_table = depth_table[order]
depth_table.to_latex("out/figures/depth_table.tex")

# Number of wins and tied wins
rank_table = result_table.rank(axis=1, method="min", ascending=False)
# print(rank_table)
wins_df = (rank_table == 1).sum(axis=0)
print(wins_df)

# Average rank
mean_rank_df = rank_table.mean(axis=0)
sem_rank_df = rank_table.sem(axis=0)
print(mean_rank_df)

# Number of timeouts
timeouts_df = result_df.groupby("Algorithm")["Timeout"].sum()
print(timeouts_df)

# Summarize aggregate scores in a table
mean_score_df = result_table.mean(axis=0)
sem_score_df = result_table.sem(axis=0)
agg_score_df = pd.concat((mean_score_df, sem_score_df, mean_rank_df, sem_rank_df, wins_df), axis=1)
agg_score_df.columns = ["Mean adversarial accuracy", "Standard error adversarial accuracy", "Mean rank", "Standard error rank", "Number of wins"]
print(agg_score_df)
agg_score_df.to_latex(figure_dir + "aggregate_scores.tex", float_format="%.3f")
