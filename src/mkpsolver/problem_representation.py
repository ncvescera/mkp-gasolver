from typing import List
import numpy as np


class MKProblem():

    def __init__(self,
                 num_f: int,
                 W: List[float],
                 items: List[List[float]],
                 sol1: List[float],
                 sol2: float,
                 num_items: None | int = None):
        self.num_f = num_f
        self.W = W
        self.items = items
        self.sol1 = sol1
        self.sol2 = sol2
        self.num_items = num_items if num_items is not None else len(items)

    @classmethod
    def from_file(cls, fpath: str, delimiter: str = ','):
        num_f = None
        W = None
        items = []
        with open(fpath, 'r') as f:
            num_f = int(f.readline().strip())
            W = np.array(f.readline().strip().split(delimiter),
                         dtype=np.float32)

            assert len(W) == num_f

            num_items = int(f.readline().strip())

            for i in range(num_items):
                tmp_item = np.array(f.readline().strip().split(delimiter),
                                    dtype=np.float32)
                assert len(tmp_item) == num_f

                items.append(tmp_item)

            sol_1 = np.array(f.readline().strip().split(delimiter),
                             dtype=np.float32)
            assert len(sol_1) == num_items

            sol_2 = float(f.readline().strip())

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
