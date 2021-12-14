from roct.upper_bound import maximum_adversarial_accuracy

import groot.datasets as datasets_module

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# Avoid type 3 fonts
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import seaborn as sns
sns.set_theme(style="whitegrid", context="paper", palette="colorblind", font_scale=1.2)

datasets = ["banknote-authentication", "blood-transfusion", "breast-cancer", "cylinder-bands", "diabetes", "haberman", "ionosphere", "wine"]
data_dir = "data/"

percentages = [0.75, 0.5, 0.25]

results = []
for dataset in datasets:
    # From the groot.datasets module, find the function with name load_<selected_dataset> then execute it
    X, y = getattr(datasets_module, f"load_{dataset.replace('-', '_')}")()[1:3]
    X = MinMaxScaler().fit_transform(X)

    for epsilon in np.linspace(0, 0.5, 51):
        Delta_l = Delta_r = np.full(X.shape[1], epsilon)
        maximum_score = maximum_adversarial_accuracy(X, y, Delta_l, Delta_r)
        results.append((dataset, epsilon, maximum_score))
    
results_df = pd.DataFrame(results, columns=["Dataset", "Epsilon", "Adversarial accuracy bound"])

_, ax = plt.subplots(figsize=(7.5, 5))
sns.lineplot(data=results_df, x="Epsilon", y="Adversarial accuracy bound", hue="Dataset", style="Dataset", markers=True, dashes=False, ax=ax)
plt.xlabel("$\epsilon$")
plt.legend(bbox_to_anchor=(0.5, 1.05), loc=8, borderaxespad=0., ncol=4)
plt.tight_layout()
plt.savefig("out/datasets_epsilon.png")
plt.savefig("out/datasets_epsilon.pdf")
plt.close()

epsilon_table = []
for dataset in datasets:
    dataset_df = results_df[results_df["Dataset"] == dataset]

    min_robustness = dataset_df["Adversarial accuracy bound"].min()
    max_robustness = dataset_df["Adversarial accuracy bound"].max()

    epsilons = []
    for percentage in percentages:
        find_bound = (max_robustness - min_robustness) * percentage + min_robustness
        index = (dataset_df["Adversarial accuracy bound"] - find_bound).abs().idxmin()
        epsilon = dataset_df.loc[index]["Epsilon"]
        epsilons.append(epsilon)
    epsilon_table.append((dataset, *epsilons))

epsilon_df = pd.DataFrame(epsilon_table, columns=["Dataset", "75%", "50%", "25%"])
epsilon_df.to_csv("out/epsilon_choices.csv")
epsilon_df.to_latex("out/epsilon_choices.tex")
