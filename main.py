import numpy as np
import argparse


class MKProblem():

    def __init__(self, num_f, W, items, sol1, sol2, num_items=None):
        self.num_f = num_f
        self.W = W
        self.items = items
        self.sol1 = sol1
        self.sol2 = sol2
        self.num_items = num_items if num_items is not None else len(items)

    @classmethod
    def from_file(cls, fpath):
        num_f = None
        W = None
        items = []
        with open(fpath, 'r') as f:
            num_f = int(f.readline().strip())
            W = f.readline().strip().split(',')

            assert len(W) == num_f

            num_items = int(f.readline().strip())

            for i in range(num_items):
                tmp_item = f.readline().strip().split(',')
                assert len(tmp_item) == num_f

                items.append(tmp_item)

            sol_1 = f.readline().strip().split(',')
            assert len(sol_1) == num_items

            sol_2 = int(f.readline().strip())

        return MKProblem(num_f=num_f,
                         W=W,
                         items=items,
                         sol1=sol_1,
                         sol2=sol_2,
                         num_items=num_items)

    def __str__(self):

        def repr_items():
            result = "ITEMS:\n"
            for item in self.items:
                result += f"   - {item}\n"

            result = result[:-1]
            return result

        return ("-- MKProblem --\n\n"
                f"NUM FEATURES: {self.num_f}\n"
                f"Ws: {self.W}\n"
                f"NUM ITEMS: {self.num_items}\n"
                f"{repr_items()}\n"
                f"SOL1: {self.sol1}\n"
                f"SOL2: {self.sol2}")


def main(args):
    problem = MKProblem.from_file(args.path)
    print(problem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Multidimensional Knapsack Problem Solver')

    parser.add_argument('path', type=str, help='Instance File Path')
    args = parser.parse_args()

    main(args)
