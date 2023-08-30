import numpy as np
import argparse
from mkpsolver.problem_representation import MKProblem
from mkpsolver.solver import GeneticAlgorithm


def main(args):
    problem = MKProblem.from_file(args.path)
    print(problem)

    # TEST: objective function
    print(problem.objective_function(np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])))

    solver = GeneticAlgorithm(problem, num_gen=100)
    solution = solver.solve()
    solution_items = problem.df[solution["best"] == 1]

    print("\n### SOLUTION ###")
    for key, value in solution.items():
        print(f"{key}: {value}")

    print("Selected Items:")
    print(solution_items)
    print(" ", solution_items.sum().to_numpy())

    # solver.init_population()

    # print(len(solver.population), solver.population)
    # print(solver.f_obj)
    # print(solver.select_mating_pool())
    # print(solver.do_crossover(np.array([1, 1, 0, 1]), np.array([0, 0, 1, 0])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multidimensional Knapsack Problem Solver')

    parser.add_argument('path', type=str, help='Instance File Path')
    args = parser.parse_args()

    main(args)
