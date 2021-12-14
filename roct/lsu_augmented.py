from pysat.examples.lsu import LSU

import time

class LSUAugmented(LSU, object):
    """ LSU with the ability to apply user-defined variable polarities to bootstrap the LSU algorithm. """

    def __init__(self, formula, solver='g4', ext_model=[], verbose=0, record_progress=False, expect_interrupt=False):
        super(LSUAugmented, self).__init__(formula, solver=solver, expect_interrupt=expect_interrupt, 
            verbose=verbose)

        if ext_model:
            # here we set user-defined polarities for some of  (or all) the variables;
            # if the polarities represent an actual model, the solver will return it in the next SAT call
            self.oracle.set_phases(literals=ext_model)

        self.record_progress = record_progress
        self.expect_interrupt = expect_interrupt
        self.upper_bounds_ = []
        self.runtimes_ = []

    def found_optimum(self):
        oracle_status = self.oracle.get_status()
        if oracle_status is None:
            return False

        return not self.oracle.get_status()

    def solve(self):
        """
            Computes a solution to the MaxSAT problem. The method implements
            the LSU/LSUS algorithm, i.e. it represents a loop, each iteration
            of which calls a SAT oracle on the working MaxSAT formula and
            refines the upper bound on the MaxSAT cost until the formula
            becomes unsatisfiable.
            Returns ``True`` if the hard part of the MaxSAT formula is
            satisfiable, i.e. if there is a MaxSAT solution, and ``False``
            otherwise.
            :rtype: bool
        """

        is_sat = False
        start_time = time.time()

        while self.oracle.solve_limited(expect_interrupt=self.expect_interrupt):
            is_sat = True
            self.model = self.oracle.get_model()
            self.cost = self._get_model_cost(self.formula, self.model)
            if self.verbose:
                print('o {0} {1}'.format(self.cost, time.time() - start_time), flush=True)
            if self.record_progress:
                self.runtimes_.append(time.time() - start_time)
                self.upper_bounds_.append(self.cost)
            if self.cost == 0:      # if cost is 0, then model is an optimum solution
                break
            self._assert_lt(self.cost)

        if is_sat:
            self.model = filter(lambda l: abs(l) <= self.formula.nv, self.model)
            if self.verbose:
                if self.found_optimum():
                    print('s OPTIMUM FOUND')
                else:
                    print('s SATISFIABLE')
        elif self.verbose:
            print('s UNSATISFIABLE')

        return is_sat