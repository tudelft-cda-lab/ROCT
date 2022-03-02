import argparse

from groot.datasets import epsilon_attacker
from groot.model import GrootTreeClassifier
from groot.toolbox import Model
from groot.treant import RobustDecisionTree
from groot.util import convert_numpy

from roct.maxsat import SATOptimalRobustTree
from roct.milp import OptimalRobustTree, BinaryOptimalRobustTree

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import numpy as np

import json

import signal


# Define exception and handler to timeout functions https://stackoverflow.com/a/25027182/15406859
class TimeoutException(Exception):  # Custom exception class
    pass


def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException()


signal.signal(signal.SIGALRM, timeout_handler)


def fit_sklearn_tree(depth, X, y):
    # Sklearn decision trees need depth to be greater than 0
    if depth == 0:
        depth = 1
        
    tree = DecisionTreeClassifier(max_depth=depth, random_state=1)
    tree.fit(X, y)
    return Model.from_sklearn(tree), False


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
        max_depth=depth, attack_model=attack_model, lsu=True, lsu_timeout=args.timeout
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
    print(Model.from_groot(groot_tree).adversarial_accuracy(X, y, attack="tree", epsilon=args.epsilon))
    print(groot_tree.to_string())

    tree = OptimalRobustTree(
        max_depth=depth,
        attack_model=attack_model,
        time_limit=args.timeout,
        warm_start_tree=groot_tree,
        cpus=1,
    )
    tree.fit(X, y)
    print(tree.to_string())
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
    "tree",
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
    help=f"Dataset to train / test on ({', '.join(datasets)})",
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
    default="out/results/",
    type=str,
    help="Directory to output results in (default out/results/)",
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
    "--fix_depth",
    default=None,
    type=int,
    help="Fix tree depth to a certain value (default None)",
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

if args.fix_depth is None:
    # Train all tree depths and pick best one according to validation adversarial accuracy
    validation_scores = []
    validation_optimality = []
    validation_models = []
    best_adv_accuracy = 0
    for depth in range(args.min_depth, args.max_depth + 1):
        total_adv_accuracy = 0
        split_optimality = []
        split_models = []

        skf = StratifiedKFold(n_splits=args.n_splits)
        for train_index, test_index in skf.split(X_train, y_train):
            X_train_cv, X_val_cv = X_train[train_index], X_train[test_index]
            y_train_cv, y_val_cv = y_train[train_index], y_train[test_index]

            if args.algorithm == "tree":
                model, optimal = fit_sklearn_tree(depth, X_train_cv, y_train_cv)
            elif args.algorithm == "groot":
                model, optimal = fit_groot(depth, X_train_cv, y_train_cv)
            elif args.algorithm == "treant":
                model, optimal = fit_treant(depth, X_train_cv, y_train_cv)
            elif args.algorithm == "lsu-maxsat":
                model, optimal = fit_maxsat_lsu(depth, X_train_cv, y_train_cv)
            elif args.algorithm == "rc2-maxsat":
                model, optimal = fit_maxsat_rc2(depth, X_train_cv, y_train_cv)
            elif args.algorithm == "milp":
                model, optimal = fit_milp(depth, X_train_cv, y_train_cv)
            elif args.algorithm == "milp-warm":
                model, optimal = fit_milp_warm(depth, X_train_cv, y_train_cv)
            elif args.algorithm == "bin-milp":
                model, optimal = fit_bin_milp(depth, X_train_cv, y_train_cv)
            elif args.algorithm == "bin-milp-warm":
                model, optimal = fit_bin_milp_warm(depth, X_train_cv, y_train_cv)

            adv_accuracy = model.adversarial_accuracy(
                X_val_cv, y_val_cv, attack="tree", epsilon=args.epsilon
            )
            total_adv_accuracy += adv_accuracy
            split_optimality.append(optimal)
            split_models.append(model.json_model)

        avg_adv_accuracy = total_adv_accuracy / args.n_splits
        validation_scores.append(avg_adv_accuracy)
        validation_optimality.append(split_optimality)
        validation_models.append(split_models)
        print(depth, avg_adv_accuracy)
        if avg_adv_accuracy > best_adv_accuracy:
            best_adv_accuracy = avg_adv_accuracy
            best_depth = depth
else:
    best_depth = args.fix_depth

# Train a model on the best cross validation depth
if args.algorithm == "tree":
    model, optimal = fit_sklearn_tree(depth, X_train, y_train)
elif args.algorithm == "groot":
    model, optimal = fit_groot(best_depth, X_train, y_train)
elif args.algorithm == "treant":
    model, optimal = fit_treant(best_depth, X_train, y_train)
elif args.algorithm == "lsu-maxsat":
    model, optimal = fit_maxsat_lsu(best_depth, X_train, y_train)
elif args.algorithm == "rc2-maxsat":
    model, optimal = fit_maxsat_rc2(best_depth, X_train, y_train)
elif args.algorithm == "milp":
    model, optimal = fit_milp(best_depth, X_train, y_train)
elif args.algorithm == "milp-warm":
    model, optimal = fit_milp_warm(best_depth, X_train, y_train)
elif args.algorithm == "bin-milp":
    model, optimal = fit_bin_milp(best_depth, X_train, y_train)
elif args.algorithm == "bin-milp-warm":
    model, optimal = fit_bin_milp_warm(best_depth, X_train, y_train)

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
if args.fix_depth is None:
    filename = f"{args.output_dir}{args.dataset}_{args.algorithm}_{args.epsilon}.json"
    results = {}
    results["train_accuracy"] = train_accuracy
    results["test_accuracy"] = test_accuracy
    results["train_adv_accuracy"] = train_adv_accuracy
    results["test_adv_accuracy"] = test_adv_accuracy
    results["best_depth"] = best_depth
    results["optimal"] = optimal
    results["validation_scores"] = validation_scores
    results["validation_optimality"] = validation_optimality
    results["validation_models"] = validation_models
    results["model"] = model.json_model
else:
    filename = f"{args.output_dir}depth_{args.fix_depth}/{args.dataset}_{args.algorithm}_{args.epsilon}.json"
    results = {}
    results["train_accuracy"] = train_accuracy
    results["test_accuracy"] = test_accuracy
    results["train_adv_accuracy"] = train_adv_accuracy
    results["test_adv_accuracy"] = test_adv_accuracy
    results["best_depth"] = best_depth
    results["optimal"] = optimal
    results["model"] = model.json_model

with open(filename, "w") as file:
    json.dump(results, file, indent=2, default=convert_numpy)
