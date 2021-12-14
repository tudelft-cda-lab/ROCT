import numpy as np

from scipy.spatial.distance import pdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

def samples_in_range(X, y, Delta_l, Delta_r):
    """
    Returns a list of tuples (i, j) where sample i (of class 0) is in
    range of sample j (of class 1)
    """
    Delta_l = np.array(Delta_l)
    Delta_r = np.array(Delta_r)
    in_range = []
    for i, (sample, label) in enumerate(zip(X, y)):
        sample_l = sample - Delta_l
        sample_r = sample + Delta_r

        for j in range(i + 1, len(X)):
            if label == y[j]:
                continue

            other_sample = X[j]
            if np.all((other_sample + Delta_r > sample_l) & (other_sample - Delta_l <= sample_r)):
                if label == 0:
                    in_range.append((i, j))
                else:
                    in_range.append((j, i))
    return in_range

def samples_in_range_linf(X, y, epsilon):
    """
    Returns a list of tuples (i, j) where sample i (of class 0) is in
    range of sample j (of class 1). Only applicable for L-inf norm.
    """
    distances = pdist(X, 'chebyshev')
    n_samples = len(X)
    reachable_distance = 2 * epsilon  # Two moving samples can reach the same point in space if they are within 2 * epsilon

    in_range = []
    for i, label in enumerate(y):
        for j in range(i + 1, len(X)):
            if label == y[j]:
                continue

            if distances[n_samples * i + j - ((i + 2) * (i + 1)) // 2] <= reachable_distance:
                if label == 0:
                    in_range.append((i, j))
                else:
                    in_range.append((j, i))
    return in_range

def maximum_adversarial_accuracy(X, y, Delta_l, Delta_r):
    i_0 = np.where(y == 0)[0]
    i_1 = np.where(y == 1)[0]

    sample_i_mapping = np.zeros(len(y), dtype=int)
    sample_i_mapping[i_0] = np.arange(len(i_0))
    sample_i_mapping[i_1] = np.arange(len(i_1))

    if np.all(Delta_l == Delta_r) and np.all(Delta_l == Delta_l[0]):
        in_range = np.array(samples_in_range_linf(X, y, Delta_l[0]))
    else:
        in_range = np.array(samples_in_range(X, y, Delta_l, Delta_r))
    
    if len(in_range) == 0:
        return 1.0

    row = sample_i_mapping[in_range[:, 0]]
    col = sample_i_mapping[in_range[:, 1]]
    data = np.ones(row.shape[0])

    graph = csr_matrix((data, (row, col)))
    matching = maximum_bipartite_matching(graph)

    n_errors = np.sum(matching != -1)

    return (X.shape[0] - n_errors) / X.shape[0]
