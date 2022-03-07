import time

import numpy as np

from groot.model import NumericalNode, Node, _TREE_LEAF, _TREE_UNDEFINED
from groot.adversary import DecisionTreeAdversary

from sklearn.utils import check_random_state

from pysat.examples.rc2 import RC2Stratified
from pysat.examples.fm import FM
from pysat.formula import WCNF, WCNFPlus

from .base import BaseOptimalRobustTree
from .lsu_augmented import LSUAugmented
from .upper_bound import samples_in_range

from threading import Timer

class SATOptimalRobustTree(BaseOptimalRobustTree):

    def __init__(self, attack_model=None, max_depth=3, time_limit=None, max_features=None, lsu=False, warm_start_tree=None, warm_start_kind="groot", record_progress=False, rc2=True, add_impossible_combinations=False, verbose=False, random_state=None):
        super().__init__(
            max_depth=max_depth,
            attack_model=attack_model,
            time_limit=time_limit,
            record_progress=record_progress,
            verbose=verbose
        )
        
        self.max_features = max_features
        self.lsu = lsu
        self.warm_start_tree = warm_start_tree
        self.warm_start_kind = warm_start_kind
        self.record_progress = record_progress
        self.rc2 = rc2
        self.add_impossible_combinations = add_impossible_combinations
        self.random_state = random_state

    def _fit_solver_specific(self, X, y):
        """
        Fit optimal robust decision tree using a MaxSAT solver.
        """
        self.random_state_ = check_random_state(self.random_state)

        self.n_samples_, self.n_features_in_ = X.shape

        self.thresholds = [self.__determine_thresholds(samples, feature) for feature, samples in enumerate(X.T)]

        weights = np.ones(X.shape[0], dtype=int)
        wcnf, variables = self.__build_sat_formula(X, y, weights)

        if self.warm_start_tree:
            warm_start = self.__generate_warm_start(X, y, variables)
        else:
            warm_start = None

        model = self.__solve_sat(wcnf, warm_start)

        self.__build_tree(model, variables)

    def __generate_warm_start(self, X, y, variables):
        tree = self.warm_start_tree
        if self.warm_start_kind == "treant":
            root = tree.to_groot_root()
        elif self.warm_start_kind == "groot":
            root = tree.root_
        else:
            raise Exception("Only tree kinds treant and groot are supported")

        assert tree.max_depth == self.max_depth

        warm_start = []
        a_vars, b_vars, c_vars, e_vars, s_vars = variables[:5]

        adversary = DecisionTreeAdversary(
            tree,
            self.warm_start_kind,
            self.attack_model,
            [True for _ in range(self.n_features_in_)],
            [None for _ in range(self.n_features_in_)],
            False,
        )

        for i, e_var in enumerate(e_vars):
            correct = adversary.adversarial_accuracy(X[i].reshape(1, -1), np.array([y[i]]))
            warm_start.append(-e_var if correct else e_var)

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

        for t in T_B:
            a = [0.0 for _ in range(self.n_features_in_)]

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

            # Negate all a SAT variables except for the chosen feature
            chosen_feature = a.index(1)
            a_warm = -a_vars[:, t-1]
            a_warm[chosen_feature] *= -1
            warm_start.extend(a_warm)

            # Turn b (threshold value) into unary encoded threshold variables
            b_index = np.searchsorted(self.thresholds[chosen_feature], b)
            for b_var in b_vars[:b_index, t-1]:
                warm_start.append(-b_var)
            for b_var in b_vars[b_index:, t-1]:
                warm_start.append(b_var)

            # Compute the s variables
            for i, sample in enumerate(X):
                can_move_left = sample[chosen_feature] - self.Delta_l[chosen_feature] <= b
                s_left_var = s_vars[i, t-1, 0]
                warm_start.append(s_left_var if can_move_left else -s_left_var)

                can_move_right = sample[chosen_feature] + self.Delta_r[chosen_feature] > b
                s_right_var = s_vars[i, t-1, 1]
                warm_start.append(s_right_var if can_move_right else -s_right_var)

        for c_var, t in zip(c_vars, T_L):
            value = nodes[t - 1].value
            warm_start.append(-c_var if value[0] >= value[1] else c_var)

        return warm_start

    def __build_sat_formula(self, X, y, weights=None):
        n, p = X.shape
        T_B = range(self.T // 2)
        T_L = range(self.T // 2, self.T)
        n_thresholds = max(len(ts) for ts in self.thresholds)

        # Create boolean variables
        counter = 0
        a = np.array([[counter := counter + 1 for m in T_B] for j in range(p)]).astype(object)
        b = np.array([[counter := counter + 1 for m in T_B] for ts in range(n_thresholds)]).astype(object)
        c = np.array([counter := counter + 1 for t in T_L]).astype(object)
        e = np.array([counter := counter + 1 for i in range(n)]).astype(object)
        s = np.array([[[counter := counter + 1 for side in range(2)] for m in T_B] for i in range(n)]).astype(object)

        # Only one feature may be chosen in each node
        if self.rc2:
            wcnf = WCNF()
        else:
            wcnf = WCNFPlus()

        if self.max_features is None:
            max_features_ = self.n_features_in_
        elif self.max_features == "sqrt":
            max_features_ = max(1, int(np.sqrt(self.n_features_in_)))
        else:
            raise Exception("Can only use values None or 'sqrt' for max_features, not " + self.max_features)

        for m in T_B:
            available_features = self.random_state_.choice(a[:, m], max_features_, replace=False)
            wcnf.append(list(available_features))

        # The thresholds have order b_{i} <= b_{i+1}
        for m in T_B:
            for ts in range(n_thresholds - 1):
                wcnf.append([-b[ts, m], b[ts + 1, m]])
        
        # Determine whether each sample can go left or right of each node
        for m in T_B:
            for j in range(p):
                # for v_index, v in enumerate(self.V[j]):
                for i in range(n):
                    thresholds_feature = self.thresholds[j]
                    delta_l = self.Delta_l[j]
                    delta_r = self.Delta_r[j]

                    # Determine s's on the left
                    l = self.__leftmost_threshold_right(thresholds_feature, X[i, j], delta_l)
                    wcnf.append([-a[j, m], b[l, m], s[i, m, 0]])

                    # Determine s's on the right
                    l = self.__rightmost_threshold_left(thresholds_feature, X[i, j], delta_r)
                    wcnf.append([-a[j, m], -b[l, m], s[i, m, 1]])

        # Count an error when a leaf in reach has the wrong label
        for i in range(n):
            for t in T_L:
                A_l, A_r = self.__ancestors(t + 1)

                clause = [e[i]]

                for m in A_l:
                    clause.append(-s[i, m - 1, 0])
                for m in A_r:
                    clause.append(-s[i, m - 1, 1])

                # Array index starts at 0 so we need to subtract len(T_B) from t
                if y[i] == 0:
                    clause.append(-c[t - len(T_B)])
                else:
                    clause.append(c[t - len(T_B)])

                wcnf.append(clause)

        # Add objective (maximize correct samples)
        for i, weight in enumerate(weights):
            wcnf.append([-e[i]], weight=weight)

        # Add constraints stating that close samples with different labels
        # cannot both be classified correctly at once.
        if self.add_impossible_combinations:
            in_range = samples_in_range(X, y, self.Delta_l, self.Delta_r)
            for sample_i, other_sample_i in in_range:
                wcnf.append([e[sample_i], e[other_sample_i]])

        return wcnf, (a, b, c, e, s, X, y)


    def __solve_sat(self, wcnf, warm_start):
        if self.verbose:
            print("Solving...")

        start_time = time.time()
        if warm_start is not None or self.lsu:
            if warm_start is None:
                warm_start = []
            with LSUAugmented(wcnf, solver="g4", ext_model=warm_start, verbose=self.verbose, record_progress=self.record_progress, expect_interrupt=True) as solver:
                if self.time_limit:
                    timer = Timer(self.time_limit, lambda solver: solver.interrupt(), [solver])
                    timer.start()
                
                success = solver.solve()

                if self.time_limit:
                    timer.cancel()

                if not success:
                    raise Exception("Model infeasible")

                self.optimal_ = solver.found_optimum()
                model = list(solver.get_model())

                if self.record_progress:
                    self.runtimes_ = solver.runtimes_
                    self.upper_bounds_ = solver.upper_bounds_
        else:
            if self.rc2:
                with RC2Stratified(wcnf, solver="g4", incr=True, adapt=False, exhaust=True, minz=True, verbose=int(self.verbose) * 10) as solver:
                    model = solver.compute()
                    self.optimal_ = model is not None
            else:
                with FM(wcnf, solver="g4", enc=3, verbose=self.verbose) as solver:
                    solver.compute()
                    model = solver.model
            

            if model is None:
                raise Exception("Model infeasible")
        total_time = time.time() - start_time

        if self.verbose:
            print("cost:", solver.cost)
            print("time:", total_time)

        return model
    
    def __build_tree(self, model, variables):
        a, b, c, e, s, X, y = variables

        p = self.n_features_in_
        n = self.n_samples_
        T_B = range(self.T // 2)
        T_L = range(self.T // 2, self.T)

        model = set(model)

        self.train_adversarial_accuracy_ = 1 - (sum(e[i] in model for i in range(n)) / n)

        # Create branching nodes with their feature and splitting threshold
        nodes = []
        for t in T_B:
            a_values = [a[j, t] in model for j in range(p)]
            feature = a_values.index(True)

            bs = [b[ts, t] in model for ts in range(len(self.thresholds[feature]))]

            delta_l = self.Delta_l[feature]
            delta_r = self.Delta_r[feature]

            split_values = []
            A_l, A_r = self.__ancestors(t + 1)
            for i in range(n):
                if all(s[i, path_m - 1, 0] in model for path_m in A_l) and all(s[i, path_m - 1, 1] in model for path_m in A_r):
                    split_values.append(X[i, feature] - delta_l)
                    split_values.append(X[i, feature] + delta_r)

            split_values = np.unique(split_values)

            if True in bs:
                i_1 = bs.index(True)

                if i_1 == 0:
                    threshold = -1
                else:
                    # Compute the threshold given all possible threshold values
                    # then use it to compute the actual threshold. The actual
                    # threshold ignores samples that end in a different leaf.
                    # This is important to get a maximal margin.
                    point_left = self.thresholds[feature][i_1 - 1]
                    point_right = self.thresholds[feature][i_1]
                    all_threshold_middle = (point_left + point_right) * 0.5

                    actual_right_i = np.searchsorted(split_values, all_threshold_middle)
                    if actual_right_i == 0:
                        threshold = -1
                    elif actual_right_i == len(split_values):
                        threshold = 2
                    else:
                        threshold = (split_values[actual_right_i - 1] + split_values[actual_right_i]) * 0.5
                    
                    if self.verbose:
                        print("Old threshold:", all_threshold_middle)
                        print("New threshold:", threshold)

                    threshold = all_threshold_middle
            else:
                threshold = 2

            # print(f"Node: {t} feature: {feature}, threshold: {threshold}")
            node = NumericalNode(feature, threshold, _TREE_UNDEFINED, _TREE_UNDEFINED, _TREE_UNDEFINED)
            nodes.append(node)
        
        # Create leaf nodes with their prediction values
        for t in T_L:
            prediction = float(c[t - len(T_B)] in model)
            value = np.array([1 - prediction, prediction])
            leaf = Node(_TREE_UNDEFINED, _TREE_LEAF, _TREE_LEAF, value)
            nodes.append(leaf)

        # Hook up the nodes to each other
        for t in T_B:
            t += 1
            node = nodes[t - 1]
            node.left_child = nodes[(t * 2) - 1]
            node.right_child = nodes[t * 2]

        self.root_ = nodes[0]

    def __determine_thresholds(self, samples, feature):
        delta_l = self.Delta_l[feature]
        delta_r = self.Delta_r[feature]

        points = np.concatenate((samples - delta_l, samples + delta_r))
        points = np.unique(np.sort(points)[:-1])

        return points

    def __rightmost_threshold_left(self, thresholds, point, delta_r):
        return np.abs(thresholds - (point + delta_r)).argmin()

    def __leftmost_threshold_right(self, thresholds, point, delta_l):
        return np.abs(thresholds - (point - delta_l)).argmin()

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
