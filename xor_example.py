from groot.adversary import DecisionTreeAdversary
from groot.datasets import epsilon_attacker
from groot.model import GrootTree
from groot.treant import RobustDecisionTree
from groot.visualization import plot_adversary

from roct.maxsat import SATOptimalRobustTree

from sklearn.preprocessing import MinMaxScaler

import numpy as np

import matplotlib.pyplot as plt

from itertools import product
from functools import reduce

import seaborn as sns

sns.set(context="paper", style="whitegrid", palette="colorblind")

def block_of_samples(locs_min, block_width, steps, dimensions, label):
    new_samples = list(product(*(np.linspace(loc_min, loc_min + block_width, steps) for loc_min in locs_min)))
    new_labels = [label] * (steps ** dimensions)
    return new_samples, new_labels

def create_dataset(block1_start=0.15, block2_start=0.65, block_width=0.2, steps=5, dimensions=2):
    blocks = []
    for blocks_i in product((0, 1), repeat=dimensions):
        xor = reduce(lambda a, b: a ^ b, blocks_i)
        block_starts = [block1_start if val == 0 else block2_start for val in blocks_i]
        blocks.append(block_of_samples(block_starts, block_width, steps, dimensions, xor))

    samples = []
    labels = []
    for new_samples, new_labels in blocks:
        samples.extend(new_samples)
        labels.extend(new_labels)

    return np.array(samples), np.array(labels)

print("Creating dataset...")
epsilon = 0.1
max_depth = 2

colors = {
    0: sns.color_palette()[0],
    1: sns.color_palette()[1],
}

X, y = create_dataset(dimensions=2)
X = MinMaxScaler().fit_transform(X)

X_0 = X[y == 0]
X_1 = X[y == 1]
plt.scatter(X_0[:, 0], X_0[:, 1], c=sns.color_palette()[0])
plt.scatter(X_1[:, 0], X_1[:, 1], c=sns.color_palette()[1])
plt.savefig("out/bad_data_2d.png")

print("Fitting GROOT...")
attack_model = [epsilon] * X.shape[1]
groot_tree = GrootTree(attack_model=attack_model, max_depth=max_depth)
groot_tree.fit(X, y)

adversary = DecisionTreeAdversary(groot_tree, "groot", attack_model)
plot_adversary(X, y, adversary, colors=colors)
plt.tight_layout(pad=0)
plt.savefig("out/bad_data_2d_groot.png")
plt.savefig("out/bad_data_2d_groot.pdf")
plt.close()

print("Fitting TREANT...")
attacker = epsilon_attacker(X.shape[1], epsilon, max_depth)
treant_tree = RobustDecisionTree(attacker=attacker, max_depth=max_depth, min_instances_per_node=2, affine=False, seed=0)
treant_tree.fit(X, y)

adversary = DecisionTreeAdversary(treant_tree, "treant", attack_model, [True for _ in range(X.shape[1])], [None for _ in range(X.shape[1])], one_adversarial_class=False)
plot_adversary(X, y, adversary, colors=colors)
plt.tight_layout(pad=0)
plt.savefig("out/bad_data_2d_treant.png")
plt.savefig("out/bad_data_2d_treant.pdf")
plt.close()

print("Fitting ROCT...")
attack_model = [epsilon] * X.shape[1]
roct_tree = SATOptimalRobustTree(attack_model=attack_model, max_depth=max_depth)
roct_tree.fit(X, y)

adversary = DecisionTreeAdversary(roct_tree, "groot", attack_model, [True for _ in range(X.shape[1])], [None for _ in range(X.shape[1])], one_adversarial_class=False)
plot_adversary(X, y, adversary, colors=colors)
plt.tight_layout(pad=0)
plt.savefig("out/bad_data_2d_roct.png")
plt.savefig("out/bad_data_2d_roct.pdf")
plt.close()
