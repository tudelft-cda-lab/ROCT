import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.2)

from roct.milp import OptimalRobustTree, BinaryOptimalRobustTree
from roct.maxsat import SATOptimalRobustTree

from groot.model import GrootTreeClassifier

# Avoid type 3 fonts
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def compute_average_runtime_cost(many_runtimes, many_costs):
    all_runtimes = [item for sublist in many_runtimes for item in sublist]
    all_runtimes = np.sort(np.unique(all_runtimes))

    costs_resampled = []
    for runtimes, costs in zip(many_runtimes, many_costs):
        indices = np.searchsorted(runtimes, all_runtimes, side='right') - 1
        costs = np.array(costs)
        costs_resampled.append(costs[indices])

    mean_costs = np.sum(costs_resampled, axis=0) / len(many_runtimes)
    sem_costs = np.std(costs_resampled, axis=0, ddof=1) / np.sqrt(len(many_runtimes))
    return all_runtimes, mean_costs, sem_costs

def plot_runtimes_cost(many_runtimes, many_costs, color_index, label, only_avg=False):
    mean_runtimes, mean_costs, sem_costs = compute_average_runtime_cost(
        many_runtimes, many_costs
    )
    if only_avg:
        plt.fill_between(mean_runtimes, mean_costs, mean_costs + sem_costs, color=sns.color_palette()[color_index], alpha=0.05)
        plt.fill_between(mean_runtimes, mean_costs, mean_costs - sem_costs, color=sns.color_palette()[color_index], alpha=0.05)
    else:
        for runtimes, costs in zip(many_runtimes, many_costs):
            plt.plot(
                runtimes, costs, drawstyle="steps-post", c=sns.color_palette()[color_index], alpha=0.2
            )
    plt.plot(mean_runtimes, mean_costs, c=sns.color_palette()[color_index], drawstyle="steps-post", label=label)
    

depth = 3
time_limit = 60
use_cached = False

data_dir = "data/"
figure_dir = "out/figures/"
output_dir = "out/"

algorithms = [
    "groot",
    "treant",
    "lsu-maxsat",
    "rc2-maxsat",
    "milp",
    "milp-warm",
    "bin-milp",
    "bin-milp-warm",
]
datasets = [
    "banknote-authentication",
    "blood-transfusion",
    "breast-cancer",
    "cylinder-bands",
    "diabetes",
    "haberman",
    "ionosphere",
    "wine",
]
epsilons = {
    "banknote-authentication": [0.07, 0.09, 0.11],
    "blood-transfusion": [0.01, 0.02, 0.03],
    "breast-cancer": [0.28, 0.39, 0.45],
    "cylinder-bands": [0.23, 0.28, 0.45],
    "diabetes": [0.05, 0.07, 0.09],
    "haberman": [0.02, 0.03, 0.05],
    "ionosphere": [0.2, 0.28, 0.36],
    "wine": [0.02, 0.03, 0.04],
}

if use_cached:
    with open(output_dir + "progress.txt") as file:
        milp_runtimes = eval(file.readline())
        milp_costs = eval(file.readline())

        bin_milp_runtimes = eval(file.readline())
        bin_milp_costs = eval(file.readline())

        milp_warm_runtimes = eval(file.readline())
        milp_warm_costs = eval(file.readline())

        bin_milp_warm_runtimes = eval(file.readline())
        bin_milp_warm_costs = eval(file.readline())

        lsu_sat_runtimes = eval(file.readline())
        lsu_sat_costs = eval(file.readline())
else:
    milp_runtimes = []
    milp_costs = []

    bin_milp_runtimes = []
    bin_milp_costs = []

    milp_warm_runtimes = []
    milp_warm_costs = []

    bin_milp_warm_runtimes = []
    bin_milp_warm_costs = []

    lsu_sat_runtimes = []
    lsu_sat_costs = []

    for dataset in datasets:
        # Load dataset samples
        X_train = np.load(data_dir + f"X_train_{dataset}.npy")
        X_test = np.load(data_dir + f"X_test_{dataset}.npy")

        # Load dataset labels
        y_train = np.load(data_dir + f"y_train_{dataset}.npy")
        y_test = np.load(data_dir + f"y_test_{dataset}.npy")

        epsilon = epsilons[dataset][1]
        attack_model = [epsilon] * X_train.shape[1]

        # MILP
        tree = OptimalRobustTree(
            attack_model=attack_model,
            max_depth=depth,
            time_limit=time_limit,
            record_progress=True,
        )
        tree.fit(X_train, y_train)
        milp_runtimes.append([0.0] + tree.runtimes_)
        milp_costs.append([1.0] + [cost / len(X_train) for cost in tree.upper_bounds_])

        # Binary-MILP
        tree = BinaryOptimalRobustTree(
            attack_model=attack_model,
            max_depth=depth,
            time_limit=time_limit,
            record_progress=True,
        )
        tree.fit(X_train, y_train)
        bin_milp_runtimes.append([0.0] + tree.runtimes_)
        bin_milp_costs.append([1.0] + [cost / len(X_train) if cost / len(X_train) <= 1.0 else 1.0 for cost in tree.upper_bounds_])

        # MILP-warm
        groot_tree = GrootTreeClassifier(
            max_depth=depth, attack_model=attack_model, min_samples_split=2, random_state=1
        )
        groot_tree.fit(X_train, y_train)
        tree = OptimalRobustTree(
            attack_model=attack_model,
            max_depth=depth,
            time_limit=time_limit,
            record_progress=True,
            warm_start_tree=groot_tree,
        )
        tree.fit(X_train, y_train)
        milp_warm_runtimes.append([0.0] + tree.runtimes_)
        milp_warm_costs.append([1.0] + [cost / len(X_train) for cost in tree.upper_bounds_])

        # Binary-MILP-warm
        tree = BinaryOptimalRobustTree(
            attack_model=attack_model,
            max_depth=depth,
            time_limit=time_limit,
            record_progress=True,
            warm_start_tree=groot_tree,
        )
        tree.fit(X_train, y_train)
        bin_milp_warm_runtimes.append([0.0] + tree.runtimes_)
        bin_milp_warm_costs.append([1.0] + [cost / len(X_train) if cost / len(X_train) <= 1.0 else 1.0 for cost in tree.upper_bounds_])

        # LSU-SAT
        tree = SATOptimalRobustTree(
            attack_model=attack_model,
            max_depth=depth,
            record_progress=True,
            lsu=True,
            time_limit=time_limit,
        )
        tree.fit(X_train, y_train)
        lsu_sat_runtimes.append([0.0] + tree.runtimes_ + [max(time_limit, tree.runtimes_[-1])])
        lsu_sat_costs.append([1.0] + [cost / len(X_train) for cost in tree.upper_bounds_] + [tree.upper_bounds_[-1] / len(X_train)])

    with open(output_dir + "progress.txt", "w") as file:
        file.write(str(milp_runtimes) + '\n')
        file.write(str(milp_costs) + '\n')

        file.write(str(bin_milp_runtimes) + '\n')
        file.write(str(bin_milp_costs) + '\n')

        file.write(str(milp_warm_runtimes) + '\n')
        file.write(str(milp_warm_costs) + '\n')

        file.write(str(bin_milp_warm_runtimes) + '\n')
        file.write(str(bin_milp_warm_costs) + '\n')

        file.write(str(lsu_sat_runtimes) + '\n')
        file.write(str(lsu_sat_costs) + '\n')

plot_runtimes_cost(milp_runtimes, milp_costs, 0, "MILP")
plot_runtimes_cost(bin_milp_runtimes, bin_milp_costs, 1, "Binary-MILP")
plot_runtimes_cost(milp_warm_runtimes, milp_warm_costs, 2, "MILP-warm")
plot_runtimes_cost(bin_milp_warm_runtimes, bin_milp_warm_costs, 3, "Binary-MILP-warm")
plot_runtimes_cost(lsu_sat_runtimes, lsu_sat_costs, 4, "LSU-MaxSAT")

plt.xlim(0.1, time_limit)
plt.xlabel("Time (s)")
plt.ylabel("% training error")
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(figure_dir + "cost_over_time.png")
plt.savefig(figure_dir + "cost_over_time.pdf")
plt.close()

plot_runtimes_cost(milp_runtimes, milp_costs, 0, "MILP", only_avg=True)
plot_runtimes_cost(bin_milp_runtimes, bin_milp_costs, 1, "Binary-MILP", only_avg=True)
plot_runtimes_cost(milp_warm_runtimes, milp_warm_costs, 2, "MILP-warm", only_avg=True)
plot_runtimes_cost(bin_milp_warm_runtimes, bin_milp_warm_costs, 3, "Binary-MILP-warm", only_avg=True)
plot_runtimes_cost(lsu_sat_runtimes, lsu_sat_costs, 4, "LSU-MaxSAT", only_avg=True)

plt.xlim(0.1, time_limit)
plt.xlabel("Time (s)")
plt.ylabel("Mean % training error")
plt.xscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(figure_dir + "mean_cost_over_time.png")
plt.savefig(figure_dir + "mean_cost_over_time.pdf")
plt.close()
