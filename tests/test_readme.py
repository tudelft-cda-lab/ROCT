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

    assert accuracy > 0.5
    assert adversarial_accuracy > 0.5
    assert adversarial_accuracy <= accuracy
