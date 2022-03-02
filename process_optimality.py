import pandas as pd

import numpy as np

from roct.upper_bound import maximum_adversarial_accuracy

import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1)

# Avoid type 3 fonts
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from tqdm import tqdm

import os

import json

optimal_dir = "out/single_results/"
data_dir = "data/"
figure_dir = "out/figures/"

results = []
for filename in tqdm(os.listdir(optimal_dir)):
    if not filename.endswith(".json"):
        continue

    with open(optimal_dir + filename) as file:
        result = json.load(file)

    dataset, algorithm, epsilon, depth = filename[:-5].split("_")
    
    if result["optimal"]:
        # Load datasets
        X_train = np.load(data_dir + f"X_train_{dataset}.npy")
        y_train = np.load(data_dir + f"y_train_{dataset}.npy")

        # Determine adversarial accuracy bound
        Delta_l = Delta_r = np.full(X_train.shape[1], fill_value=float(epsilon))
        adv_acc_bound = maximum_adversarial_accuracy(X_train, y_train, Delta_l, Delta_r)
    
        results.append((dataset, dataset + "-" + epsilon, algorithm, depth, result["runtime"], result["train_adv_accuracy"], result["test_adv_accuracy"], adv_acc_bound))

columns = ["Dataset", "Dataset-Epsilon", "Algorithm", "Depth", "Runtime", "Optimal score", "Optimal test score", "Adversarial accuracy bound"]
result_df = pd.DataFrame(results, columns=columns)

groot_results = []
for filename in os.listdir(optimal_dir):
    if not filename.endswith(".json"):
        continue

    with open(optimal_dir + filename) as file:
        result = json.load(file)

    dataset, algorithm, epsilon, depth = filename[:-5].split("_")

    if algorithm == "groot":
        groot_results.append((dataset + "-" + epsilon, depth, result["train_adv_accuracy"]))

columns = ["Dataset-Epsilon", "Depth", "Train adversarial accuracy"]
groot_result_df = pd.DataFrame(groot_results, columns=columns)

all_results_df = result_df.join(groot_result_df.set_index(["Dataset-Epsilon", "Depth"]), on=["Dataset-Epsilon", "Depth"])

all_results_df["diffs"] = all_results_df["Train adversarial accuracy"] / all_results_df["Optimal score"]
all_results_df["Ratio"] = "GROOT / Optimal"
bound_results_df = all_results_df.copy()
bound_results_df["diffs"] = bound_results_df["Optimal score"] / bound_results_df["Adversarial accuracy bound"]
bound_results_df["Ratio"] = "Optimal / Bound"
all_results_df = pd.concat((all_results_df, bound_results_df))
ax = plt.subplots(figsize=(5, 2.5))
sns.swarmplot(x="Depth", y="diffs", data=all_results_df, hue="Ratio", order=["1","2","3","4","5"], size=3, dodge=True)
plt.ylabel("Ratio")
plt.tight_layout(pad=0.1)
plt.savefig(figure_dir + "diffs.png")
plt.savefig(figure_dir + "diffs.pdf")
plt.close()
