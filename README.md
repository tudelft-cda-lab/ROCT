# ROCT: Robust Optimal Classification Trees

This folder contains the scripts and code needed to reproduce the results of 'Robust Optimal Classification Trees Against Adversarial Examples'. Much of our code extends the code retrieved from 'GROOT' at https://github.com/tudelft-cda-lab/GROOT .

## Installation instructions
The code needs a new version of python, at least 3.7. We recommend using virtual environments to install the necessary packages with pip:
```
pip install -r requirements.txt
```

## Simple example
Below is a small example for running ROCT (using MILP or MaxSAT solver) on a toy dataset using the Scikit-learn API.

```python3
from roct.milp import OptimalRobustTree, BinaryOptimalRobustTree
from roct.maxsat import SATOptimalRobustTree

from groot.toolbox import Model

from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
X, y = make_moons(noise=0.3, random_state=0)
X_test, y_test = make_moons(noise=0.3, random_state=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Define the attacker's capabilities (L-inf norm radius 0.3)
epsilon = 0.1
attack_model = [epsilon, epsilon]

names = ("MILP", "Binary MILP", "MaxSAT")
trees = (
    OptimalRobustTree(max_depth=2, attack_model=attack_model),
    BinaryOptimalRobustTree(max_depth=2, attack_model=attack_model),
    SATOptimalRobustTree(max_depth=2, attack_model=attack_model),
)
for name, tree in zip(names, trees):
    tree.fit(X, y)

    # Determine the accuracy and adversarial accuracy against attackers
    accuracy = tree.score(X_test, y_test)
    model = Model.from_groot(tree)
    adversarial_accuracy = model.adversarial_accuracy(X_test, y_test, attack="tree", epsilon=epsilon)

    print(name)
    print("Accuracy:", accuracy)
    print("Adversarial Accuracy:", adversarial_accuracy)
    print()
```

## Reproducing results
Our figures and fitted trees can be accessed under the `out/` directory, but the results can also be generated from scratch. Fitting all trees and running all experiments can take many days depending on how many parallel cores are available.

### Downloading the datasets
Running the script `download_datasets.py` will download all required datasets from openML into the `data/` directory (which has already been done).

### Main results
The main script for fitting and scoring trees is `run.py`, which can be accessed by a command line interface. It uses 3-fold cross validation to set the `max_depth` hyperparameter, trains a tree with the best setting and tests it performance on the test set. To use the script to train and score all trees one can run the commands in `all_jobs.txt`, the `parallel` GNU command is particularly useful here. For example, the following will run the `run.py` script in parallel on 15 cores:
```
parallel -j 15 :::: all_jobs.txt
```
The resulting trees will generate under `out/results/`. To generate figures and tables please run `python process_results.py` after the parallel process has finished.

### Optimality experiment
To fit trees for our optimality experiment we have a similar procedure but it trains trees for 2 hours instead of 30 minutes (per tree). Please run:
```
parallel -j 15 :::: all_jobs_single.txt
```
The resulting trees will generate under `out/results_single_7200_5/`. To generate figures please run `python process_optimality.py` after the parallel process has finished.

### Bound (theorem 2) and choosing epsilon
The script for computing the bounds while varying epsilon is `choose_epsilons.py`. In the resulting table `out/epsilon_choices` we only change the last column of `blood-transfusion` in the paper to avoid duplicate entries.

### Solver progress over time
The script for plotting solver progress over time is `performance_over_time.py`. This script runs each early-stoppable solver one after the other with trees of depth 3 on one epsilon setting per dataset. This script does not run in parallel so it can take many hours to run.

### Short scripts
The XOR dataset, trees and images can be created using `xor_example.py`, a summary of the used datasets using `summarize_datasets.py`.
