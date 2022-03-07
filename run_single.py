import argparse

from groot.datasets import epsilon_attacker
from groot.model import GrootTreeClassifier
from groot.toolbox import Model
from groot.treant import RobustDecisionTree
from groot.util import convert_numpy

from roct.maxsat import SATOptimalRobustTree
from roct.milp import OptimalRobustTree, BinaryOptimalRobustTree

import numpy as np

import json

import time

import signal


# Define exception and handler to timeout functions https://stackoverflow.com/a/25027182/15406859
class TimeoutException(Exception):  # Custom exception class
    pass


def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException()


signal.signal(signal.SIGALRM, timeout_handler)


def fit_groot(depth, X, y):
    attack_model = [args.epsilon] * X.shape[1]
    tree = GrootTreeClassifier(
        max_depth=depth, attack_model=attack_model, min_samples_split=2, random_state=1
    )
    tree.fit(X, y)
    return Model.from_groot(tree), False


def fit_treant(depth, X, y):
    attacker = epsilon_attacker(X.shape[1], args.epsilon, depth)
    tree = RobustDecisionTree(
        max_depth=depth,
        affine=False,
        seed=0,
        min_instances_per_node=2,
        attacker=attacker,
    )

    # Set an alarm for args.timeout seconds
    signal.alarm(args.timeout)

    try:
        # Try to fit the tree within args.timeout seconds
        tree.fit(X, y)
        return Model.from_treant(tree), False
    except TimeoutException:
        # If timed out, train a 0 depth dummy tree and return it
        print("Timeout!")
        tree = RobustDecisionTree(
            max_depth=0,
            affine=False,
            seed=0,
            min_instances_per_node=2,
            attacker=attacker,
        )
        tree.fit(X, y)
        return Model.from_treant(tree), False


def fit_maxsat_lsu(depth, X, y):
    attack_model = [args.epsilon] * X.shape[1]
    tree = SATOptimalRobustTree(
        max_depth=depth, attack_model=attack_model, lsu=True, time_limit=args.timeout
    )
    tree.fit(X, y)
    return Model.from_groot(tree), tree.optimal_


def fit_maxsat_rc2(depth, X, y):
    attack_model = [args.epsilon] * X.shape[1]
    tree = SATOptimalRobustTree(max_depth=depth, attack_model=attack_model, rc2=True)

    # Set an alarm for args.timeout seconds
    signal.alarm(args.timeout), tree.optimal_

    try:
        # Try to fit the tree within args.timeout seconds
        tree.fit(X, y)
        return Model.from_groot(tree), tree.optimal_
    except TimeoutException:
        # If timed out, train a 0 depth dummy tree and return it
        print("Timeout!")
        tree = SATOptimalRobustTree(max_depth=0, attack_model=attack_model, rc2=True)
        tree.fit(X, y)
        return Model.from_groot(tree), False


def fit_milp(depth, X, y):
    attack_model = [args.epsilon] * X.shape[1]
    tree = OptimalRobustTree(
        max_depth=depth, attack_model=attack_model, time_limit=args.timeout, cpus=1
    )
    tree.fit(X, y)
    return Model.from_groot(tree), tree.optimal_


def fit_milp_warm(depth, X, y):
    attack_model = [args.epsilon] * X.shape[1]

    groot_tree = GrootTreeClassifier(
        max_depth=depth, attack_model=attack_model, min_samples_split=2, random_state=1
    )
    groot_tree.fit(X, y)

    tree = OptimalRobustTree(
        max_depth=depth,
        attack_model=attack_model,
        time_limit=args.timeout,
        warm_start_tree=groot_tree,
        cpus=1,
    )
    tree.fit(X, y)
    return Model.from_groot(tree), tree.optimal_


def fit_bin_milp(depth, X, y):
    attack_model = [args.epsilon] * X.shape[1]
    tree = BinaryOptimalRobustTree(
        max_depth=depth, attack_model=attack_model, time_limit=args.timeout, cpus=1
    )
    tree.fit(X, y)
    return Model.from_groot(tree), tree.optimal_


def fit_bin_milp_warm(depth, X, y):
    attack_model = [args.epsilon] * X.shape[1]

    groot_tree = GrootTreeClassifier(
        max_depth=depth, attack_model=attack_model, min_samples_split=2, random_state=1
    )
    groot_tree.fit(X, y)

    tree = BinaryOptimalRobustTree(
        max_depth=depth,
        attack_model=attack_model,
        time_limit=args.timeout,
        warm_start_tree=groot_tree,
        cpus=1,
    )
    tree.fit(X, y)
    return Model.from_groot(tree), tree.optimal_


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

parser = argparse.ArgumentParser(description="Fit and evaluate a robust decision tree")

parser.add_argument(
    "algorithm",
    type=str,
    help=f"Name of the decision tree learning algorithm ({', '.join(algorithms)})",
)
parser.add_argument(
    "dataset",
    type=str,
    help="Dataset to train / test on",
)
parser.add_argument(
    "depth",
    type=int,
    help="Maximum depth of the decision tree",
)
parser.add_argument(
    "-e",
    "--epsilon",
    default=0.1,
    type=float,
    help="L-infinity epsilon radius for adversarial examples (default 0.1)",
)
parser.add_argument(
    "-t",
    "--timeout",
    default=None,
    type=int,
    help="Time limit in seconds (default None)",
)
parser.add_argument(
    "-d",
    "--data_dir",
    default="data/",
    type=str,
    help="Directory containing the datasets in .npy format (default data/)",
)
parser.add_argument(
    "-o",
    "--output_dir",
    default="out/single_results/",
    type=str,
    help="Directory to output results in (default out/single_results/)",
)
parser.add_argument(
    "--min_depth",
    default=0,
    type=int,
    help="Minimum tree depth to try (default 0)",
)
parser.add_argument(
    "--max_depth",
    default=4,
    type=int,
    help="Maximum tree depth to try (default 4)",
)
parser.add_argument(
    "--n_splits",
    default=3,
    type=int,
    help="Number of stratified K-fold splits for tree depth selection (default 3)",
)

args = parser.parse_args()

# Check if algorithm is supported
if args.algorithm not in algorithms:
    raise ValueError(
        f"Algorithm '{args.algorithm}' is not supported, must be one of ({','.join(algorithms)})"
    )

# Check if dataset is supported
if args.dataset not in datasets:
    raise ValueError(
        f"Dataset '{args.dataset}' is not supported, must be one of ({','.join(datasets)})"
    )

# Load dataset samples
X_train = np.load(args.data_dir + f"X_train_{args.dataset}.npy")
X_test = np.load(args.data_dir + f"X_test_{args.dataset}.npy")

# Load dataset labels
y_train = np.load(args.data_dir + f"y_train_{args.dataset}.npy")
y_test = np.load(args.data_dir + f"y_test_{args.dataset}.npy")

# First run GROOT once to get rid of the JIT compilation overhead
if args.algorithm == "groot":
    GrootTreeClassifier(max_depth=1, attack_model=[0.1] * X_train.shape[1]).fit(X_train, y_train)

# Train a model on the given depth
start_time = time.time()
if args.algorithm == "groot":
    model, optimal = fit_groot(args.depth, X_train, y_train)
elif args.algorithm == "treant":
    model, optimal = fit_treant(args.depth, X_train, y_train)
elif args.algorithm == "lsu-maxsat":
    model, optimal = fit_maxsat_lsu(args.depth, X_train, y_train)
elif args.algorithm == "rc2-maxsat":
    model, optimal = fit_maxsat_rc2(args.depth, X_train, y_train)
elif args.algorithm == "milp":
    model, optimal = fit_milp(args.depth, X_train, y_train)
elif args.algorithm == "milp-warm":
    model, optimal = fit_milp_warm(args.depth, X_train, y_train)
elif args.algorithm == "bin-milp":
    model, optimal = fit_bin_milp(args.depth, X_train, y_train)
elif args.algorithm == "bin-milp-warm":
    model, optimal = fit_bin_milp_warm(args.depth, X_train, y_train)
runtime = time.time() - start_time

# Compute accuracy scores
train_accuracy = model.accuracy(X_train, y_train)
test_accuracy = model.accuracy(X_test, y_test)
print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)

# Compute robustness scores
train_adv_accuracy = model.adversarial_accuracy(X_train, y_train, attack="tree", epsilon=args.epsilon)
test_adv_accuracy = model.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=args.epsilon)
print("Train adversarial accuracy:", train_adv_accuracy)
print("Test adversarial accuracy:", test_adv_accuracy)

# Record experiment results and save as JSON
filename = f"{args.output_dir}/{args.dataset}_{args.algorithm}_{args.epsilon}_{args.depth}.json"
results = {}
results["train_accuracy"] = train_accuracy
results["test_accuracy"] = test_accuracy
results["train_adv_accuracy"] = train_adv_accuracy
results["test_adv_accuracy"] = test_adv_accuracy
results["optimal"] = optimal
results["runtime"] = runtime
results["model"] = model.json_model
with open(filename, "w") as file:
    json.dump(results, file, indent=2, default=convert_numpy)
