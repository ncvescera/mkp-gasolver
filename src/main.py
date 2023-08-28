import numpy as np
import argparse
from mkpsolver.problem_representation import MKProblem
from mkpsolver.solver import GeneticAlgorithm


def main(args):
    problem = MKProblem.from_file(args.path)
    print(problem)

    # TEST: objective function
    print(problem.objective_function(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])))

    solver = GeneticAlgorithm(problem)
    solver.init_population()
    print(solver.population)
    # print(solver.do_crossover(np.array([1, 1, 0, 1]), np.array([0, 0, 1, 0])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multidimensional Knapsack Problem Solver')

    parser.add_argument('path', type=str, help='Instance File Path')
    args = parser.parse_args()

    main(args)
