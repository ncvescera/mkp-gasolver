import numpy as np
import argparse
from mkpsolver.problem_representation import MKProblem


def main(args):
    problem = MKProblem.from_file(args.path)
    print(problem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multidimensional Knapsack Problem Solver')

    parser.add_argument('path', type=str, help='Instance File Path')
    args = parser.parse_args()

    main(args)
