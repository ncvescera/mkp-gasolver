import numpy as np
import argparse
from mkpsolver.problem_representation import MKProblem


class GeneticAlgorithm():
    # Gli elementi vengono rappresentati con 0 NON LO PRENDO, 1 LO PRENDO
    def __init__(
            self,
            problem: MKProblem,
            # num_elem=None,
            num_gen=100,
            pcross=.9,
            pmut=.01):

        self.problem = problem
        self.num_items = problem.get_dim()
        self.pcross = pcross
        self.pmut = pmut
        self.num_gen = num_gen

    def init_population(self):
        self.population = []
        self.f_obj = np.zeros(self.num_items)
        self.best = None
        self.best_f = 1e300

        for i in range(self.num_items):
            ind = 0


def main(args):
    problem = MKProblem.from_file(args.path)
    print(problem)

    print(problem.objective_function(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multidimensional Knapsack Problem Solver')

    parser.add_argument('path', type=str, help='Instance File Path')
    args = parser.parse_args()

    main(args)
