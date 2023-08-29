import numpy as np
from mkpsolver.typing import List, Tuple, Solution
from mkpsolver.problem_representation import MKProblem
import random


class GeneticAlgorithm():
    # Gli elementi vengono rappresentati con 0 NON LO PRENDO, 1 LO PRENDO
    def __init__(
            self,
            problem: MKProblem,  # TODO: si può omettere ?
            num_elem=16,
            num_gen=100,
            pcross=.9,
            pmut=.01):

        self.problem = problem
        self.num_items = problem.get_dim()
        self.num_elem = num_elem
        self.pcross = pcross
        self.pmut = pmut
        self.num_gen = num_gen

    def solve(self):
        self.improvements = []
        self.init_population()

        # print(self.population)

        for gen in range(1, self.num_gen + 1):
            mating_pool = self.select_mating_pool()
            children = self.do_crossover(mating_pool)
            self.do_mutation(children)
            self.select_new_population(children, gen)

        return None, None, None  # TODO: completare

    def init_population(self):
        self.population = []
        # self.f_obj = np.zeros(self.num_elem)
        self.f_obj = list(np.zeros(self.num_elem))
        self.best = None
        self.best_f = float('-inf')  # huge number

        for i in range(self.num_elem):
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
            self.update_best(tmp_sol, self.f_obj[i], 0)

    def update_best(self, x: Solution, fx: float, gen: int):
        # maximize the objective function
        if fx > self.best_f:
            self.best_f = fx
            self.best = x
            print(f"new best {fx} @ gent: {gen}")
            self.improvements.append((
                gen,
                fx,
            ))

    def select_mating_pool(self) -> List[Tuple[Solution, Solution]]:

        def roulette_wheel():
            raise NotImplementedError

        def tournament(k: int = 5) -> Solution:
            random_select_solutions = [
                random.randint(0,
                               len(self.population) - 1) for _ in range(k)
            ]

            # generate a dictioray {solution index : solution fitness}
            # e.g. solution 1/16 has fitness vale of 287
            selected_objectivefunctions = {
                i: self.f_obj[i]
                for i in random_select_solutions
            }

            # find solution index with max fitless value
            # e.g. solution 3/16 has the max fintess value
            max_index = max(selected_objectivefunctions,
                            key=selected_objectivefunctions.get)

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

    def do_mutation(self, children: List[Solution]):
        '''
        Randomly flip bits according to pmut probability
        '''
        for child in children:  # TODO: ricontrollare
            for i in range(len(child)):
                if random.random() < self.pmut:
                    child[i] = int(not child[i])

    # TODO: compleatre
    def repair_operator(self) -> Solution:
        raise NotImplementedError

    def select_new_population(self, children: List[Solution], gen: int):

        def select_best():
            total_solutions: List[Solution] = self.population + children
            total_fintesses: List[float] = self.f_obj + [
                self.problem.objective_function(c) for c in children
            ]

            assert len(total_solutions) == len(total_fintesses)

            total_indexes: List[int] = list(range(len(total_solutions)))
            total_indexes.sort(key=lambda i: total_fintesses[i], reverse=True)
            best_indexes: List[int] = total_indexes[:self.num_elem]

            self.population = [total_solutions[i] for i in best_indexes]
            self.f_obj = [total_fintesses[i] for i in best_indexes]

            self.update_best(self.population[0], self.f_obj[0], gen)

        select_best()
