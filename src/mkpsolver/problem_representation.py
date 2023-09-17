from mkpsolver.typing import List, Tuple, Solution
import numpy as np
import pandas as pd
import logging as lg


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
        self.df = pd.DataFrame(np.array(items)).transpose()
        self.df.columns = [f"f{i}" for i in range(len(self.W))]

        sorted_df = self.df.copy()

        self.df = pd.concat(
            [self.df, pd.DataFrame(self.values, columns=['Value'])], axis=1)

        # Generating sorted indexes by Object Importance
        object_importance_values = []
        for i in range(self.num_items):
            tmp_importance = (sorted_df.iloc[i] *
                              self.W).sum() / self.W.sum() / self.values[i]
            object_importance_values.append(tmp_importance)

        sorted_df = pd.concat([
            sorted_df,
            pd.DataFrame(object_importance_values, columns=['Importance'])
        ],
                              axis=1)

        lg.debug(f"IMPORTANCE:\n{sorted_df}")

        self.sorted_value_objects_indexes = sorted_df.sort_values(
            by='Importance', ascending=True).index.to_numpy()

        lg.debug(f"SORTED INDEXES: {self.sorted_value_objects_indexes}")

    def get_dim(self) -> int:
        return self.num_items

    def objective_function(self, x: Solution) -> float:
        return self.df[x == 1]['Value'].sum()

    @classmethod
    def from_file(cls, fpath: str, delimiter: str = ','):
        num_f = None
        W = None  # upper constraint limits
        items = []

        with open(fpath, 'r') as f:
            num_items = int(f.readline().strip())
            values = np.array(f.readline().strip().split(delimiter),
                              dtype=np.float32)

            # make sure upper constriant limits match the num features len
            assert len(values) == num_items

            num_f = int(f.readline().strip())

            for i in range(num_f):
                tmp_item = np.array(f.readline().strip().split(delimiter),
                                    dtype=np.float32)

                # make shure item features match the num features len
                assert len(tmp_item) == num_items

                items.append(tmp_item)

            W = np.array(f.readline().strip().split(delimiter),
                         dtype=np.float32)

            # make shure values match the item number
            assert len(W) == num_f

            sol = float(f.readline().strip())

        return MKProblem(num_f=num_f,
                         W=W,
                         items=items,
                         values=values,
                         sol2=sol,
                         num_items=num_items)

    def __str__(self):

        return ("-- MKProblem --\n\n"
                f"NUM FEATURES: {self.num_f}\n"
                f"Ws: {self.W}\n"
                f"NUM ITEMS: {self.num_items}\n"
                f"ITEMS:\n{self.df}\n"
                f"VALUES: {self.values}\n"
                f"SOL2: {self.sol2}")
