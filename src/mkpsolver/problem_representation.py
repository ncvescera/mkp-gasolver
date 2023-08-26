from typing import List
import numpy as np
import pandas as pd


class MKProblem():

    def __init__(self,
                 num_f: int,
                 W: List[float],
                 items: List[List[float]],
                 values: List[float],
                 sol2: float,
                 num_items: None | int = None):
        self.num_f = num_f
        self.W = W
        self.items = items
        self.values = values
        self.sol2 = sol2
        self.num_items = num_items if num_items is not None else len(items)

        # generating DataFrame using a ndarry (item represented as matrix)
        self.df = pd.DataFrame(np.array(items),
                               columns=[f"f{i}" for i in range(len(self.W))])
        self.df = pd.concat(
            [self.df, pd.DataFrame(self.values, columns=['Values'])], axis=1)

    def get_dim(self):
        return self.num_items

    def objective_function(self, x):
        # x è una soluzione con 0 elemento non preso, 1 preso
        return self.df[x == 1]['Values'].sum()

    @classmethod
    def from_file(cls, fpath: str, delimiter: str = ','):
        num_f = None
        W = None  # upper constraint limits
        items = []

        with open(fpath, 'r') as f:
            num_f = int(f.readline().strip())
            W = np.array(f.readline().strip().split(delimiter),
                         dtype=np.float32)

            # make shure upper constriant limits match the num features len
            assert len(W) == num_f

            num_items = int(f.readline().strip())

            for i in range(num_items):
                tmp_item = np.array(f.readline().strip().split(delimiter),
                                    dtype=np.float32)

                # make shure item features match the num features len
                assert len(tmp_item) == num_f

                items.append(tmp_item)

            values = np.array(f.readline().strip().split(delimiter),
                              dtype=np.float32)

            # make shure values match the item number
            assert len(values) == num_items

            sol_2 = float(f.readline().strip())

        return MKProblem(num_f=num_f,
                         W=W,
                         items=items,
                         values=values,
                         sol2=sol_2,
                         num_items=num_items)

    def __str__(self):

        return ("-- MKProblem --\n\n"
                f"NUM FEATURES: {self.num_f}\n"
                f"Ws: {self.W}\n"
                f"NUM ITEMS: {self.num_items}\n"
                f"ITEMS:\n{self.df}\n"
                f"VALUES: {self.values}\n"
                f"SOL2: {self.sol2}")
