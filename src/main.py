import argparse
from mkpsolver.problem_representation import MKProblem
from mkpsolver.solver import GeneticAlgorithm
import logging


def main(args):
    problem = MKProblem.from_file(args.path)
    print(problem)

    solver = GeneticAlgorithm(problem,
                              pcross=args.crossover_probability,
                              pmut=args.mutation_probability,
                              num_elem=args.population_lenght,
                              num_gen=args.number_generation,
                              tk=args.tournament_k)
    solution = solver.solve()
    solution_items = problem.df[solution["best"] == 1]

    print("\n### SOLUTION ###")
    for key, value in solution.items():
        print(f"{key}: {value}")

    print("Selected Items:")
    print(solution_items)
    print(" ", solution_items.sum().to_numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multidimensional Knapsack Problem Solver')

    parser.add_argument('path', type=str, help='Instance File Path')
    parser.add_argument('-plen',
                        '--population_lenght',
                        default=16,
                        type=int,
                        help='Initial Population Lenght')
    parser.add_argument('-pcross',
                        '--crossover_probability',
                        default=.9,
                        type=float,
                        help='Crossover Probability (from 0 to 1)')
    parser.add_argument('-pmut',
                        '--mutation_probability',
                        default=.01,
                        type=float,
                        help='Mutation probability (from 0 to 1)')
    parser.add_argument('-ngen',
                        '--number_generation',
                        default=100,
                        type=int,
                        help='Number of generations')
    parser.add_argument('-tk',
                        '--tournament_k',
                        default=5,
                        type=int,
                        help='Tournament random solution to select')
    parser.add_argument('-log',
                        '--log_level',
                        default="WARNINGS",
                        type=str,
                        choices=['DEBUG', 'INFO', 'WARNINGS'],
                        help='Logging Level')

    args = parser.parse_args()
    logging.basicConfig(encoding='utf-8',
                        level=getattr(logging, args.log_level, None))

    main(args)
