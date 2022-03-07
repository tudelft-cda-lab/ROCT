from numpy.core.fromnumeric import var

import json

from numbers import Number

import math

import numpy as np

from groot.adversary import DecisionTreeAdversary
from groot.model import GrootTreeClassifier, NumericalNode, Node, _TREE_LEAF, _TREE_UNDEFINED
from groot.util import convert_numpy

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array

import gurobipy as gp
from gurobipy import GRB

class OptimalRobustTree(BaseEstimator, ClassifierMixin):

    def __init__(self, attack_model=None, max_depth=3, time_limit=None, warm_start_tree=None, warm_start_kind="groot", cpus=1, record_progress=False, verbose=False):
        self.attack_model = attack_model
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.warm_start_tree = warm_start_tree
        self.warm_start_kind = warm_start_kind
        self.cpus = cpus
        self.record_progress = record_progress
        self.verbose = verbose

        # We say that an untrained tree is not optimal
        self.optimal_ = False

    def __parse_attack_model(self, attack_model):
        Delta_l = []
        Delta_r = []
        for attack in attack_model:
            if isinstance(attack, Number):
                Delta_l.append(attack)
                Delta_r.append(attack)
            elif isinstance(attack, tuple):
                Delta_l.append(attack[0])
                Delta_r.append(attack[1])
            elif attack == "":
                Delta_l.append(0)
                Delta_r.append(0)
            elif attack == ">":
                Delta_l.append(0)
                Delta_r.append(1)
            elif attack == "<":
                Delta_l.append(1)
                Delta_r.append(0)
            elif attack == "<>":
                Delta_l.append(1)
                Delta_r.append(1)
        return Delta_l, Delta_r

    def fit(self, X, y):
        """
        Fit optimal robust decision tree using a MILP solver. MILP formulation
        closely resembles that of Bertsimas and Dunn (Optimal Decision Trees).
        """
        self.n_samples_, self.n_features_ = X.shape

        if self.attack_model is None:
            self.attack_model = [0.0] * X.shape[1]
        
        self.Delta_l, self.Delta_r = self.__parse_attack_model(self.attack_model)
        self.T = (2 ** (self.max_depth + 1)) - 1

        self.thresholds = [self.__determine_thresholds(samples, feature) for feature, samples in enumerate(X.T)]

        warm_start = self.__generate_warm_start(X, y, self.warm_start_tree)

        model, variables, tolerance = self.__build_model_gurobi(X, y)
        
        self.__solve_model_gurobi(model, variables, warm_start, tolerance)

    def __normalize_thresholds(self, root):
        queue = [root]
        while queue:
            node = queue.pop()
            if node.is_leaf():
                continue

            if node.threshold > 1.0:
                node.threshold = 1.0
            elif node.threshold < 0.0:
                node.threshold = 0.0

            queue.insert(0, node.left_child)
            queue.insert(0, node.right_child)

    def __compute_s_warm_start(self, a_warm, b_warm, X):
        s_warm = {}
        for i, sample in enumerate(X):
            for t, (a, b) in enumerate(zip(a_warm, b_warm)):
                s_left = np.dot(sample - self.Delta_l, a) <= b
                s_right = np.dot(sample + self.Delta_r, a) > b

                s_warm[(i + 1, t + 1, 0)] = float(s_left)
                s_warm[(i + 1, t + 1, 1)] = float(s_right)
        return s_warm

    def __compute_z_warm_start(self, s_warm):
        z_warm = {}
        for i in range(1, self.n_samples_ + 1):
            for t in range((self.T // 2) + 1, self.T + 1):
                A_l, A_r = self.__ancestors(t)
                z = all(s_warm[(i, m, 0)] for m in A_l) and all(s_warm[(i, m, 1)] for m in A_r)
                z_warm[i, t] = float(z)
        return z_warm

    def __generate_warm_start(self, X, y, tree):
        if tree is None:
            return

        if self.warm_start_kind == "treant":
            root = tree.to_groot_root()
        elif self.warm_start_kind == "groot":
            root = tree.root_
        else:
            raise Exception("Only tree kinds treant and groot are supported")

        # Force all thresholds in range [0, 1]
        self.__normalize_thresholds(root)

        adversary = DecisionTreeAdversary(
            tree,
            self.warm_start_kind,
            self.attack_model,
            [True for _ in range(self.n_features_)],
            [None for _ in range(self.n_features_)],
            False,
        )

        e_warm = []
        for i in range(len(y)):
            e_warm.append(1 - adversary.adversarial_accuracy(X[i].reshape(1, -1), np.array([y[i]])))

        # We need this function instead of a simple breadth first traversal
        # for all nodes since the GROOT tree can be pruned.
        def get_node(tree, t):
            if t == 1:
                return root

            A = [t]
            while t > 1:
                t //= 2
                A.append(t)
            A = list(reversed(A))[1:]

            node = root
            for node_id in A:
                if node.is_leaf():
                    break

                if node_id % 2 == 0:
                    # Go left
                    node = node.left_child
                else:
                    # Go right
                    node = node.right_child

            if node.is_leaf() and A[-1] not in range((self.T // 2) + 1, self.T + 1):
                # If we found a leaf expecting a node return a dummy node
                return None
            
            if node_id == A[-1]:
                # If the last node_id was reached we can return the node
                return node

            if node.is_leaf() and A[-1] in range((self.T // 2) + 1, self.T + 1):
                # If we found a leaf while looking for a leaf it is correct
                return node

            # Otherwise the tree was incomplete here and we return a dummy node
            return None

        nodes = [get_node(tree, t) for t in range(1, self.T + 1)]

        T_B = range(1, (self.T // 2) + 1)
        T_L = range((self.T // 2) + 1, self.T + 1)

        a_warm = []
        b_warm = []
        for t in T_B:
            a = [0.0 for _ in range(self.n_features_)]

            node = nodes[t - 1]
            if node is None:
                # Find the first ancestor of this node that is not None
                repeat_t = t // 2
                while nodes[repeat_t - 1] is None:
                    repeat_t //= 2
                
                new_node = nodes[repeat_t - 1]

                if new_node.is_leaf():
                    # If the node is a leaf then send all samples left
                    a[0] = 1.0
                    b = 1.0
                else:
                    # If the node is not a leaf repeat that node's values
                    a[new_node.feature] = 1.0
                    b = new_node.threshold
            else:
                if node.is_leaf():
                    # If the node is a leaf then send all samples left
                    a[0] = 1.0
                    b = 1.0
                else:
                    # If the node is not a leaf repeat that node's values
                    a[node.feature] = 1.0
                    b = node.threshold
                

            if b < 0.0:
                b = 0.0
            elif b > 1.0:
                b = 1.0

            a_warm.append(a)
            b_warm.append(b)

        c_warm = []
        for t in T_L:
            value = nodes[t - 1].value
            c_warm.append(0 if value[0] >= value[1] else 1)

        s_warm = self.__compute_s_warm_start(a_warm, b_warm, X)
        z_warm = self.__compute_z_warm_start(s_warm)

        return a_warm, b_warm, c_warm, e_warm, s_warm, z_warm

    def __determine_epsilon(self, X):
        best_epsilon = 1.0
        for feature in range(self.n_features_):
            values = np.concatenate((
                X[:, feature],
                X[:, feature] - self.Delta_l[feature],
                X[:, feature] + self.Delta_r[feature],
            ))
            values = np.sort(values)
            differences = np.diff(values)

            # Determine the smallest non-zero difference
            epsilon = np.min(differences[np.nonzero(differences)])
            if epsilon < best_epsilon:
                best_epsilon = epsilon

        if best_epsilon > 1e-04:
            best_epsilon = 1e-04

        if best_epsilon < 1e-08:
            best_epsilon = 1e-08
        if self.verbose:
            print('epsilon:', best_epsilon)
        return best_epsilon

    def __build_model_gurobi(self, X, y):
        p = self.n_features_
        n = self.n_samples_
        T_B = range(1, (self.T // 2) + 1)
        T_L = range((self.T // 2) + 1, self.T + 1)
        Delta_l = self.Delta_l
        Delta_r = self.Delta_r

        model = gp.Model("Optimal_Robust_Tree_Fitting")

        a = model.addVars(range(1, p + 1), T_B, vtype=GRB.BINARY, name="a")
        z = model.addVars(range(1, n + 1), T_L, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="z")
        s = model.addVars(range(1, n + 1), T_B, range(2), vtype=GRB.BINARY, name="s")
        b = model.addVars(T_B, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="b")
        c = model.addVars(T_L, vtype=GRB.BINARY, name="c")
        e = model.addVars(range(1, n + 1), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="e")

        # The objective is to minimize the sum of errors (weighted by 1 each)
        model.setObjective(e.sum(), GRB.MINIMIZE)

        # Let the nodes only split on one feature
        for t in T_B:
            model.addConstr(gp.quicksum(a[j, t] for j in range(1, p + 1)) == 1)

        epsilon = self.__determine_epsilon(X)
        self.epsilon_ = epsilon
        M_left = M_right = 2 + epsilon
        for i in range(1, n + 1):
            for m in T_B:
                model.addConstr(gp.quicksum(a[j, m] * X[i-1, j-1] for j in range(1, p + 1)) - gp.quicksum(a[j, m] * Delta_l[j-1] for j in range(1, p + 1)) >= epsilon + b[m] - M_left * s[i, m, 0])
                model.addConstr(gp.quicksum(a[j, m] * X[i-1, j-1] for j in range(1, p + 1)) + gp.quicksum(a[j, m] * Delta_r[j-1] for j in range(1, p + 1)) <= b[m] + M_right * s[i, m, 1])

        for i in range(1, n + 1):
            for t in T_L:
                A_l, A_r = self.__ancestors(t)

                model.addConstr(gp.quicksum(s[i, m, 0] for m in A_l) + gp.quicksum(s[i, m, 1] for m in A_r) - len(A_l + A_r) + 1 <= z[i, t])

        for i in range(1, n + 1):
            for t in T_L:
                if y[i - 1] == 0:
                    model.addConstr(z[i, t] + c[t] - 1 <= e[i])
                else:
                    model.addConstr(z[i, t] - c[t] <= e[i])

        tolerance = 10 ** (int(math.log10(epsilon)) - 1)
        return model, (a, z, e, s, b, c), tolerance

    def __solve_model_gurobi(self, model, variables, warm_start, tolerance):
        a, z, e, s, b, c = variables

        p = self.n_features_
        n = self.n_samples_
        T_B = range(1, (self.T // 2) + 1)
        T_L = range((self.T // 2) + 1, self.T + 1)

        if warm_start:
            a_warm, b_warm, c_warm, e_warm, s_warm, z_warm = warm_start

            for j in range(1, p + 1):
                for t in T_B:
                    a[j, t].start = a_warm[t - 1][j - 1]  # a_warm's indices are reversed

            for t in T_B:
                b[t].start = b_warm[t - 1]

            for i, t in enumerate(T_L):
                label = c_warm[i]
                c[t].start = label

            for i in range(1, n + 1):
                e[i].start = e_warm[i - 1]

            for key in s_warm:
                i, t, side = key
                s[i, t, side].start = s_warm[key]
            
            for key in z_warm:
                i, t = key
                z[i, t].start = z_warm[key]

        model.write('model.lp')

        output_flag = 1 if self.verbose else 0
        options = [
            ("OutputFlag", output_flag),
            ('IntFeasTol', tolerance),
            ('MIPGap', tolerance),
            ('Presolve', 2),
            ('MIPFocus', 1),
            ('Cuts', 2),
            ('Method', 0),
        ]
        if self.time_limit:
            options.append(('TimeLimit', self.time_limit))
        if self.cpus:
            options.append(('Threads', self.cpus))

        for option in options:
            model.setParam(*option)

        # If record progress is True then keep track of the lower and upper
        # bounds over time
        if self.record_progress:
            self.lower_bounds_ = []
            self.upper_bounds_ = []
            self.runtimes_ = []

            def callback(model, where):
                if where == GRB.Callback.MIP:
                    self.upper_bounds_.append(model.cbGet(GRB.Callback.MIP_OBJBST))
                    self.lower_bounds_.append(model.cbGet(GRB.Callback.MIP_OBJBND))
                    self.runtimes_.append(model.cbGet(GRB.Callback.RUNTIME))

            model.optimize(callback)
        else:
            model.optimize()

        if self.verbose:
            print("Error:", sum([e[i].X for i in range(1, n + 1)]))

        for t in T_B:
            if self.verbose:
                print(b[t].X)

        # Create branching nodes with their feature and splitting threshold
        nodes = []
        for t in T_B:
            a_values = [a[j, t].X for j in range(1, p + 1)]

            if 1 in a_values:
                feature = a_values.index(1)

                candidates = self.thresholds[feature]
                i = np.abs(candidates - b[t].X).argmin()

                if math.isclose(b[t].X, candidates[i]):
                    # threshold = candidates[i]
                    if i == len(candidates) - 1:
                        # Prevent out of bounds
                        threshold = candidates[i] + self.epsilon_
                    else:
                        threshold = (candidates[i] + candidates[i + 1]) * 0.5
                elif b[t].X < candidates[i]:
                    if i == 0:
                        # Prevent out of bounds
                        threshold = candidates[i] - self.epsilon_
                        # threshold = -1.0
                    else:
                        threshold = (candidates[i-1] + candidates[i]) * 0.5
                else:
                    if i == len(candidates) - 1:
                        # Prevent out of bounds
                        threshold = candidates[i] + self.epsilon_
                        # threshold = 2.0
                    else:
                        threshold = (candidates[i] + candidates[i + 1]) * 0.5
            else:
                # If there is no a_j == 1 then this node is a dummy that should
                # not apply a split. Threshold = 1 enforces this.
                feature = 0
                threshold = 2.0

            if self.verbose:
                print(f"Node: {t} feature: {feature}, threshold: {threshold}")
            node = NumericalNode(feature, threshold, _TREE_UNDEFINED, _TREE_UNDEFINED, _TREE_UNDEFINED)
            nodes.append(node)

        
        # Create leaf nodes with their prediction values
        for t in T_L:
            value = np.array([round(1 - c[t].X), round(c[t].X)])
            if self.verbose:
                print(f"Leaf: {t} value: {value}")
            leaf = Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)
            nodes.append(leaf)

        # Hook up the nodes to each other
        for t in T_B:
            node = nodes[t - 1]
            node.left_child = nodes[(t * 2) - 1]
            node.right_child = nodes[t * 2]

        self.root_ = nodes[0]
        self.optimal_ = model.Status == GRB.OPTIMAL

    def __determine_thresholds(self, samples, feature):
        delta_l = self.Delta_l[feature]
        delta_r = self.Delta_r[feature]

        points = np.concatenate((samples - delta_l, samples + delta_r))
        points = np.unique(np.sort(points)[:-1])

        return points

    def __ancestors(self, t: int):
        A_l = []
        A_r = []
        while t > 1:
            if t % 2 == 0:
                A_l.append(t // 2)
            else:
                A_r.append(t // 2)
            t //= 2
        return A_l, A_r

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.

        The class probability is the fraction of samples of the same class in
        the leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        proba : array of shape (n_samples,)
            The probability for each input sample of being malicious.
        """

        X = check_array(X)

        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        return np.array(predictions)

    def predict(self, X):
        """
        Predict the classes of the input samples X.

        The predicted class is the most frequently occuring class label in a 
        leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels
        """

        return np.argmax(self.predict_proba(X), axis=1)

    def to_string(self):
        result = ""
        result += f"Parameters: {self.get_params()}\n"

        if hasattr(self, "root_"):
            result += f"Tree:\n{self.root_.pretty_print()}"
        else:
            result += "Tree has not yet been fitted"

        return result

    def to_xgboost_json(self, output_file="tree.json"):
        if hasattr(self, "root_"):
            dictionary, _ = self.root_.to_xgboost_json(0, 0)
        else:
            raise Exception("Tree is not yet fitted")

        if output_file is None:
            return dictionary
        else:
            with open(output_file, "w") as fp:
                # If saving to file then surround dict in list brackets
                json.dump([dictionary], fp, indent=2, default=convert_numpy)

    def prune(self):
        bounds = np.tile(np.array([0, 1], dtype=np.float32), (self.n_features_, 1))

        for _ in range(self.max_depth):
            # Without decision nodes we do not have to prune
            if self.root_.is_leaf():
                break

            self.root_ = self.root_.prune(bounds)


class BinaryOptimalRobustTree(BaseEstimator, ClassifierMixin):

    def __init__(self, attack_model=None, max_depth=3, time_limit=None, warm_start_tree=None, verbose=False, cpus=1, record_progress=False):
        self.attack_model = attack_model
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.warm_start_tree = warm_start_tree
        self.verbose = verbose
        self.cpus = cpus
        self.record_progress = record_progress

    def __parse_attack_model(self, attack_model):
        Delta_l = []
        Delta_r = []
        for attack in attack_model:
            if isinstance(attack, Number):
                Delta_l.append(attack)
                Delta_r.append(attack)
            elif isinstance(attack, tuple):
                Delta_l.append(attack[0])
                Delta_r.append(attack[1])
            elif attack == "":
                Delta_l.append(0)
                Delta_r.append(0)
            elif attack == ">":
                Delta_l.append(0)
                Delta_r.append(1)
            elif attack == "<":
                Delta_l.append(1)
                Delta_r.append(0)
            elif attack == "<>":
                Delta_l.append(1)
                Delta_r.append(1)
        return Delta_l, Delta_r

    def fit(self, X, y):
        """
        Fit optimal robust decision tree using a MILP solver. MILP formulation
        closely resembles that of Bertsimas and Dunn (Optimal Decision Trees).
        """
        self.n_samples_, self.n_features_ = X.shape
        self.majority_class_ = np.argmax(np.bincount(y))

        if self.attack_model is None:
            self.attack_model = [0.0] * X.shape[1]
        
        self.Delta_l, self.Delta_r = self.__parse_attack_model(self.attack_model)
        self.T = (2 ** (self.max_depth + 1)) - 1

        self.thresholds = [self.__determine_thresholds(samples, feature) for feature, samples in enumerate(X.T)]
        self.V = [self.__determine_V(X[:, i]) for i in range(X.shape[1])]

        if self.warm_start_tree:
            warm_start = self.__generate_warm_start(X, y, self.warm_start_tree)
        else:
            warm_start = None

        if self.verbose:
            for t in self.thresholds:
                print(len(t))

        model, variables = self.__build_model_gurobi(X, y)

        self.__solve_model_gurobi(model, variables, warm_start)

    def __normalize_thresholds(self, tree):
        queue = [tree.root_]
        while queue:
            node = queue.pop()
            if node.is_leaf():
                continue

            if node.threshold > 1.0:
                node.threshold = 1.0
            elif node.threshold < 0.0:
                node.threshold = 0.0

            queue.insert(0, node.left_child)
            queue.insert(0, node.right_child)

    def __compute_s_warm_start(self, a_warm, b_warm, X):
        s_warm = {}
        for i, sample in enumerate(X):
            for t, (a, b) in enumerate(zip(a_warm, b_warm)):
                split_feature = a.index(1)
                left_lim = sample[split_feature] - self.Delta_l[split_feature]
                right_lim = sample[split_feature] + self.Delta_r[split_feature]

                left_index = np.searchsorted(self.thresholds[split_feature], left_lim)
                right_index = np.searchsorted(self.thresholds[split_feature], right_lim)

                if left_index == len(b):
                    left_index -= 1
                if right_index == len(b):
                    right_index -= 1
                
                s_warm[(i + 1, t + 1, 0)] = float(b[left_index] == 0)
                s_warm[(i + 1, t + 1, 1)] = float(b[right_index] == 1)
        return s_warm

    def __generate_warm_start(self, X, y, tree=None):
        if tree is None:
            tree = GrootTreeClassifier(
                max_depth=self.max_depth,
                attack_model=self.attack_model,
                one_adversarial_class=False,
                random_state=0,
            )
            tree.fit(X, y)

        # Force all thresholds in range [0, 1]
        self.__normalize_thresholds(tree)

        adversary = DecisionTreeAdversary(
            tree,
            "groot",
            self.attack_model,
            [True for _ in range(self.n_features_)],
            [True for _ in range(self.n_features_)],
            False
        )

        e_warm = []
        for i in range(len(y)):
            e_warm.append(1 - adversary.adversarial_accuracy(X[i].reshape(1, -1), np.array([y[i]])))

        # We need this function instead of a simple breadth first traversal
        # for all nodes since the GROOT tree can be pruned.
        def get_node(tree, t):
            if t == 1:
                return tree.root_

            A = [t]
            while t > 1:
                t //= 2
                A.append(t)
            A = list(reversed(A))[1:]

            node = tree.root_
            for node_id in A:
                if node.is_leaf():
                    break

                if node_id % 2 == 0:
                    # Go left
                    node = node.left_child
                else:
                    # Go right
                    node = node.right_child

            if node.is_leaf() and A[-1] not in range((self.T // 2) + 1, self.T + 1):
                # If we found a leaf expecting a node return a dummy node
                return None
            
            if node_id == A[-1]:
                # If the last node_id was reached we can return the node
                return node

            if node.is_leaf() and A[-1] in range((self.T // 2) + 1, self.T + 1):
                # If we found a leaf while looking for a leaf it is correct
                return node

            # Otherwise the tree was incomplete here and we return a dummy node
            return None

        nodes = [get_node(tree, t) for t in range(1, self.T + 1)]

        T_B = range(1, (self.T // 2) + 1)
        T_L = range((self.T // 2) + 1, self.T + 1)
        n_thresholds = max(len(thresholds) for thresholds in self.thresholds)

        a_warm = []
        b_warm = []
        for t in T_B:
            a = [0.0 for _ in range(self.n_features_)]

            node = nodes[t - 1]
            if node is None:
                # Find the first ancestor of this node that is not None and
                # simply repeat that node's values
                repeat_t = t // 2
                while nodes[repeat_t - 1] is None:
                    repeat_t //= 2
                a[nodes[repeat_t - 1].feature] = 1.0
                b = nodes[repeat_t - 1].threshold
            else:
                a[nodes[t - 1].feature] = 1.0
                b = nodes[t - 1].threshold
            a_warm.append(a)

            if b < 0.0:
                b = 0.0
            elif b > 1.0:
                b = 1.0

            # Turn b (threshold value) into unary encoded threshold variables
            chosen_feature = a.index(1)
            b_index = np.searchsorted(self.thresholds[chosen_feature], b)
            b_values = [0.0] * b_index + [1.0] * (n_thresholds - b_index)
            b_warm.append(b_values)

        c_warm = []
        for t in T_L:
            value = nodes[t - 1].value
            c_warm.append(0 if value[0] >= value[1] else 1)

        s_warm = self.__compute_s_warm_start(a_warm, b_warm, X)

        return a_warm, b_warm, c_warm, e_warm, s_warm

    def __build_model_gurobi(self, X, y):
        p = self.n_features_
        n = self.n_samples_
        T_B = range(1, (self.T // 2) + 1)
        T_L = range((self.T // 2) + 1, self.T + 1)
        n_thresholds = max(len(ts) for ts in self.thresholds)

        model = gp.Model("Optimal_Robust_Tree_Fitting")

        a = model.addVars(range(1, p + 1), T_B, vtype=GRB.BINARY, name="a")
        b = model.addVars(range(1, n_thresholds + 1), T_B, lb=0, ub=1, vtype=GRB.BINARY, name="b")
        c = model.addVars(T_L, vtype=GRB.BINARY, name="c")
        e = model.addVars(range(1, n + 1), lb=0, ub=1, vtype=GRB.CONTINUOUS, name="e")
        s = model.addVars(range(1, n + 1), T_B, range(2), vtype=GRB.CONTINUOUS, name="s")

        # The objective is to minimize the sum of errors (weighted by 1 each)
        model.setObjective(e.sum(), GRB.MINIMIZE)

        # Let the nodes only split on one feature
        for t in T_B:
            model.addConstr(gp.quicksum(a[j, t] for j in range(1, p + 1)) == 1)

        # The thresholds have order b_{i} <= b_{i+1}
        for m in T_B:
            for ts in range(1, n_thresholds):
                model.addConstr(b[ts, m] <= b[ts + 1, m])
        
        # Determine whether each sample can go left or right of each node
        for m in T_B:
            for j in range(1, p + 1):
                # for v_index, v in enumerate(self.V[j]):
                for i in range(1, n + 1):
                    thresholds_feature = self.thresholds[j - 1]
                    delta_l = self.Delta_l[j - 1]
                    delta_r = self.Delta_r[j - 1]

                    # Determine s's on the left
                    l = self.__leftmost_threshold_right(thresholds_feature, X[i-1, j-1], delta_l)
                    model.addConstr(s[i, m, 0] >= a[j, m] + (1 - b[l, m]) - 1)

                    # Determine s's on the right
                    l = self.__rightmost_threshold_left(thresholds_feature, X[i-1, j-1], delta_r)
                    model.addConstr(s[i, m, 1] >= a[j, m] + b[l, m] - 1)

        # Count an error when a leaf in reach has the wrong label
        for i in range(1, n + 1):
            for t in T_L:
                A_l, A_r = self.__ancestors(t)

                if y[i - 1] == 0:
                    model.addConstr(e[i] >= c[t] + gp.quicksum(s[i, m, 0] for m in A_l) + gp.quicksum(s[i, m, 1] for m in A_r) - len(A_l + A_r))
                else:
                    model.addConstr(e[i] >= (1 - c[t]) + gp.quicksum(s[i, m, 0] for m in A_l) + gp.quicksum(s[i, m, 1] for m in A_r) - len(A_l + A_r))
                    
        return model, (a, b, c, e, s)

    def __determine_thresholds(self, samples, feature):
        delta_l = self.Delta_l[feature]
        delta_r = self.Delta_r[feature]

        points = np.concatenate((samples - delta_l, samples + delta_r))
        # points = np.delete(points, np.where((points > 1) | (points < 0))[0])
        points = np.unique(np.sort(points)[:-1])

        return points

    def __determine_V(self, samples):
        return list(np.unique(samples))

    def __rightmost_threshold_left(self, thresholds, point, delta_r):
        return np.abs(thresholds - (point + delta_r)).argmin() + 1

    def __leftmost_threshold_right(self, thresholds, point, delta_l):
        return np.abs(thresholds - (point - delta_l)).argmin() + 1

    def __solve_model_gurobi(self, model, variables, warm_start):
        a, b, c, e, s = variables

        p = self.n_features_
        n = self.n_samples_
        T_B = range(1, (self.T // 2) + 1)
        T_L = range((self.T // 2) + 1, self.T + 1)
        n_thresholds = max(len(thresholds) for thresholds in self.thresholds)

        if warm_start:
            a_warm, b_warm, c_warm, e_warm, s_warm = warm_start

            for j in range(1, p + 1):
                for t in T_B:
                    a[j, t].start = a_warm[t - 1][j - 1]  # a_warm's indices are reversed

            for t in T_B:
                for ts in range(1, n_thresholds + 1):
                    b[ts, t].start = b_warm[t - 1][ts - 1]

            for i, t in enumerate(T_L):
                c[t].start = c_warm[i]

            for i in range(1, n + 1):
                e[i].start = e_warm[i - 1]

            for key in s_warm:
                i, t, side = key
                s[i, t, side].start = s_warm[key]

        output_flag = 1 if self.verbose else 0
        options = [
            ("OutputFlag", output_flag),
            ('Presolve', 2),
            ('MIPFocus', 1),
            ('Cuts', 2),
            ('Method', 0),
        ]
        if self.time_limit:
            options.append(('TimeLimit', self.time_limit))
        if self.cpus:
            options.append(('Threads', self.cpus))

        for option in options:
            model.setParam(*option)

        # If record progress is True then keep track of the lower and upper
        # bounds over time
        if self.record_progress:
            self.lower_bounds_ = []
            self.upper_bounds_ = []
            self.runtimes_ = []

            def callback(model, where):
                if where == GRB.Callback.MIP:
                    self.upper_bounds_.append(model.cbGet(GRB.Callback.MIP_OBJBST))
                    self.lower_bounds_.append(model.cbGet(GRB.Callback.MIP_OBJBND))
                    self.runtimes_.append(model.cbGet(GRB.Callback.RUNTIME))

            model.optimize(callback)
        else:
            model.optimize()

        # Create just a leaf if no solution was found
        if model.Status == GRB.TIME_LIMIT and model.ObjVal == float('inf'):
            self.optimal_ = False

            value = np.array([1 - self.majority_class_, self.majority_class_])
            self.root_ = Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)
            return

        # Create branching nodes with their feature and splitting threshold
        nodes = []
        for t in T_B:
            a_values = [a[j, t].X for j in range(1, p + 1)]
            if 1 in a_values:
                feature = a_values.index(1)

                bs = [b[i + 1, t].X for i in range(len(self.thresholds[feature]))]
                try:
                    i_1 = bs.index(1)

                    if i_1 == 0:
                        threshold = self.thresholds[feature][0] - 0.0001
                    else:
                        point_left = self.thresholds[feature][bs.index(1) - 1]
                        point_right = self.thresholds[feature][bs.index(1)]
                        threshold = (point_left + point_right) * 0.5
                except:
                    threshold = self.thresholds[feature][-1] + 0.0001
            else:
                # If there is no a_j == 1 then this node is a dummy that should
                # not apply a split. Threshold = 1 enforces this.
                feature = 0
                threshold = self.thresholds[feature][-1] + 0.0001

            if self.verbose:
                print(f"Node: {t} feature: {feature}, threshold: {threshold}")
            node = NumericalNode(feature, threshold, _TREE_UNDEFINED, _TREE_UNDEFINED, _TREE_UNDEFINED)
            nodes.append(node)

        
        # Create leaf nodes with their prediction values
        for t in T_L:
            value = [1 - c[t].X, c[t].X]

            if self.verbose:
                print(f"Leaf: {t} value: {value}")

            if value[0] is None or value[1] is None:
                value = [0.5, 0.5]
            
            value = np.array(value)
            leaf = Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)
            nodes.append(leaf)

        # Hook up the nodes to each other
        for t in T_B:
            node = nodes[t - 1]
            node.left_child = nodes[(t * 2) - 1]
            node.right_child = nodes[t * 2]

        self.root_ = nodes[0]
        self.optimal_ = model.Status == GRB.OPTIMAL

    def __ancestors(self, t: int):
        A_l = []
        A_r = []
        while t > 1:
            if t % 2 == 0:
                A_l.append(t // 2)
            else:
                A_r.append(t // 2)
            t //= 2
        return A_l, A_r

    def prune(self):
        bounds = np.tile(np.array([0, 1], dtype=np.float32), (self.n_features_, 1))

        for _ in range(self.max_depth):
            # Without decision nodes we do not have to prune
            if self.root_.is_leaf():
                break

            self.root_ = self.root_.prune(bounds)

    def predict_proba(self, X):
        """
        Predict class probabilities of the input samples X.

        The class probability is the fraction of samples of the same class in
        the leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        proba : array of shape (n_samples,)
            The probability for each input sample of being malicious.
        """

        X = check_array(X)

        predictions = []
        for sample in X:
            predictions.append(self.root_.predict(sample))

        return np.array(predictions)

    def predict(self, X):
        """
        Predict the classes of the input samples X.

        The predicted class is the most frequently occuring class label in a 
        leaf.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted class labels
        """

        return np.argmax(self.predict_proba(X), axis=1)

    def to_string(self):
        result = ""
        result += f"Parameters: {self.get_params()}\n"

        if hasattr(self, "root_"):
            result += f"Tree:\n{self.root_.pretty_print()}"
        else:
            result += "Tree has not yet been fitted"

        return result

    def to_xgboost_json(self, output_file="tree.json"):
        if hasattr(self, "root_"):
            dictionary, _ = self.root_.to_xgboost_json(0, 0)
        else:
            raise Exception("Tree is not yet fitted")

        if output_file is None:
            return dictionary
        else:
            with open(output_file, "w") as fp:
                # If saving to file then surround dict in list brackets
                json.dump([dictionary], fp, indent=2, default=convert_numpy)
