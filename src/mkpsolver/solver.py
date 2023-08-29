import numpy as np
from mkpsolver.typing import List, Tuple, Solution
from mkpsolver.problem_representation import MKProblem
import random


class GeneticAlgorithm():
    # Gli elementi vengono rappresentati con 0 NON LO PRENDO, 1 LO PRENDO
    def __init__(
            self,
            problem: MKProblem,  # TODO: si può omettere ?
            num_gen=100,
            pcross=.9,
            pmut=.01):

        self.problem = problem
        self.num_items = problem.get_dim()
        self.pcross = pcross
        self.pmut = pmut
        self.num_gen = num_gen

    def init_population(self, num_elem: int = 16) -> List[Solution]:
        self.population = []
        self.f_obj = np.zeros(num_elem)
        self.best = None
        self.best_f = float('inf')  # huge number

        for i in range(num_elem):
            tmp_sol = np.zeros(self.num_items)

            # list of indexes (e.g. 1 means df[1])
            T = list(range(self.num_items))

            # temporary actual constrints sum
            R = np.zeros(len(self.problem.W))

            # randomly extract an item
            j = T.pop(random.randint(0, len(T) - 1))
            item = self.problem.df.loc[:, self.problem.df.columns !=
                                       'Value'].loc[j].to_numpy()

            # try to add extracted item, then extract a new one and so on
            while all(R + item <= self.problem.W):
                tmp_sol[j] = 1
                R = R + item

                # no more items left, continue to new solution
                if len(T) <= 0:
                    break

                j = T.pop(random.randrange(len(T)))
                item = self.problem.df.loc[:, self.problem.df.columns !=
                                           'Value'].loc[j].to_numpy()

            self.population.append(tmp_sol)
            self.f_obj[i] = self.problem.objective_function(
                tmp_sol)  # TODO: usare dict ?

    def select_mating_pool(self):

        def roulette_wheel():
            raise NotImplementedError

        def tournament(k=5):
            random_select_items = [
                random.randint(0,
                               len(self.population) - 1) for _ in range(k)
            ]

            tmp = {i: self.f_obj[i] for i in random_select_items}

            max_index = max(tmp, key=tmp.get)

            return self.population[max_index]

        mating_pool = []

        for i in range(len(self.population) // 2):
            c1 = tournament()
            c2 = tournament()
            mating_pool.append((
                c1,
                c2,
            ))

        return mating_pool

    def do_crossover(
            self, mating_pool: List[Tuple[Solution,
                                          Solution]]) -> List[Solution]:

        def uniform_crossover_operator(s1: Solution, s2: Solution) -> Solution:
            '''
            From parents (s1 and s2) generate only 1 child (c) using
            a random probability to choose a chromosome from s1 or s2
            '''
            c = np.zeros(len(s1))

            for i in range(len(s1)):
                # random True or False.
                # faster than `random.choice([True, False])`
                c[i] = s1[i] if bool(random.getrandbits(1)) else s2[i]

            return c

        children = []

        for s1, s2 in mating_pool:
            if random.random() < self.pcross:
                c = uniform_crossover_operator(s1, s2)
                children.append(c)
                continue

            children.append(s1)
            children.append(s2)

        return children

    def do_mutation(self, children: List[Solution]) -> List[Solution]:
        '''
        Randomly flip bits according to pmut probability
        '''
        for child in children:  # TODO: ricontrollare
            for i in range(len(child)):
                if random.random() < self.pmut:
                    child[i] = int(not child[i])

    def repair_operator(self) -> Solution:
        raise NotImplementedError
