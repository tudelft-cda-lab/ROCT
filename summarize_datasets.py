from groot.datasets import load_all

from roct.upper_bound import maximum_adversarial_accuracy

from sklearn.preprocessing import MinMaxScaler

import numpy as np

datasets = set((
    "banknote-authentication",
    "blood-transfusion",
    "breast-cancer",
    "cylinder-bands",
    "diabetes",
    "haberman",
    "ionosphere",
    "wine",
))

epsilons = {
    "banknote-authentication": [0.07,0.09,0.11],
    "blood-transfusion": [0.01,0.02,0.03],
    "breast-cancer": [0.28,0.39,0.45],
    "cylinder-bands": [0.23,0.28,0.45],
    "diabetes": [0.05,0.07,0.09],
    "haberman": [0.02,0.03,0.05],
    "ionosphere": [0.2,0.28,0.36],
    "wine": [0.02,0.03,0.04],
}

for name, X, y in load_all():
    if name not in datasets:
        continue

    X = MinMaxScaler().fit_transform(X)

    for epsilon in epsilons[name]:
        deltas = [epsilon] * X.shape[1]
        class_counts = np.bincount(y)
        majority = class_counts[1] / len(y) if class_counts[1] > class_counts[0] else class_counts[0] / len(y)
        max_score = maximum_adversarial_accuracy(X, y, deltas, deltas)
        print(name, epsilon, *X.shape, ("%.3f" % majority).lstrip('0'), ("%.3f" % max_score).lstrip('0'), sep=" & ", end="\\\\\n")
