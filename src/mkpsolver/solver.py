import numpy as np
from mkpsolver.typing import List, Tuple, Dict, Solution
from mkpsolver.problem_representation import MKProblem
import random
import logging as lg


class GeneticAlgorithm():
    # Gli elementi vengono rappresentati con 0 NON LO PRENDO, 1 LO PRENDO
    def __init__(
            self,
            problem: MKProblem,  # TODO: si può omettere ?
            num_elem=16,
            num_gen=100,
            pcross=.9,
            pmut=.01):

        self.problem: MKProblem = problem
        self.num_items = problem.get_dim()
        self.num_elem = num_elem
        self.pcross = pcross
        self.pmut = pmut
        self.num_gen = num_gen

        lg.debug(f"SOLVER INSTANCE: num_items={self.num_items};"
                 f"num_elem={self.num_elem};"
                 f"pcross={self.pcross};"
                 f"pmut={self.pmut};"
                 f"num_gen={self.num_gen}")

    def solve(self) -> Dict[str, any]:
        self.improvements: List[Tuple[int, float]] = []

        self.init_population()

        for gen in range(1, self.num_gen + 1):
            lg.debug(f"STARTING GEN {gen}")

            mating_pool: List[Tuple[Solution,
                                    Solution]] = self.select_mating_pool()
            children: List[Solution] = self.do_crossover(mating_pool)
            self.do_mutation(children)
            self.repair_operator(children)
            self.select_new_population(children, gen)

        return {
            "best": self.best,
            "best_fitness": self.best_f,
            "improvements": self.improvements
        }

    def init_population(self):
        lg.debug("INITIAL POPULATION: Running")

        self.population: List[Solution] = []
        self.f_obj: List[float] = list(np.zeros(self.num_elem))
        self.best: Solution = None
        self.best_f: float = float('-inf')  # tiny number

        for i in range(self.num_elem):
            tmp_sol: Solution = np.zeros(self.num_items)

            # list of indexes (e.g. 1 means df[1])
            T: List[int] = list(range(self.num_items))

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
            self.f_obj[i] = self.problem.objective_function(tmp_sol)
            self.update_best(tmp_sol, self.f_obj[i], 0)

        lg.debug("INITIAL POPULATION")
        for elem in self.population:
            lg.debug(f"- {list(elem)}")

    def update_best(self, x: Solution, fx: float, gen: int):
        lg.debug("UPDATE BEST: running")

        # maximize the objective function
        if fx > self.best_f:
            self.best_f = fx
            self.best = x.copy()  # save actuale solution, not reference

            lg.info(f"new best {fx} @ gent: {gen}")
            lg.debug(f"NEW BEST: {self.best}")

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

        lg.debug("MATING POOL: Running")

        mating_pool = []

        for i in range(len(self.population) // 2):
            c1 = tournament()
            c2 = tournament()
            mating_pool.append((
                c1,
                c2,
            ))

            lg.debug(f"- {c1} {c2}")

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

        lg.debug("CROSSOVER: Running")

        children = []

        for s1, s2 in mating_pool:
            if random.random() < self.pcross:
                c = uniform_crossover_operator(s1, s2)
                children.append(c)

                lg.debug(f"{s1} x {s2} = {c}")

                continue

            children.append(s1)
            children.append(s2)

        return children

    def do_mutation(self, children: List[Solution]):
        '''
        Randomly flip bits according to pmut probability
        '''
        lg.debug("MUTATION: Running")

        for child in children:
            for i in range(len(child)):
                if random.random() < self.pmut:
                    to_log = f"{child} -> "

                    child[i] = int(not child[i])

                    to_log += f"{child}"
                    lg.debug(to_log)

    def repair_operator(self, children: List[Solution]) -> Solution:
        lg.debug("REPAIR: Running")

        sorted_value_objects_indexes = self.problem.df.sort_values(
            by='Value', ascending=True).index.to_numpy()

        lg.debug(f"sorted indexes: {sorted_value_objects_indexes}")

        for child in children:
            child_parameters = self.problem.df.loc[:,
                                                   self.problem.df.columns !=
                                                   'Value'].loc[
                                                       child ==
                                                       1].sum().to_numpy()
            # good child, check next child
            if all(child_parameters <= self.problem.W):
                continue

            old_child = child.copy()

            lg.debug(f"BAD CHILD: {old_child}")
            lg.debug(f"CP       : {child_parameters}")
            lg.debug(f"W        : {self.problem.W}")

            # DROP PHASE
            # delete high Values objects until solution is feasible
            i = len(sorted_value_objects_indexes) - 1  # last element
            while any(child_parameters > self.problem.W):
                # delete item from solution
                child[sorted_value_objects_indexes[i]] = 0

                # update parameters
                child_parameters = self.problem.df.loc[:, self.problem.df.
                                                       columns != 'Value'].loc[
                                                           child ==
                                                           1].sum().to_numpy()
                i = i - 1

            lg.debug("--- DROP PHASE ---")
            lg.debug(f"FIX CHILD: {child}")
            lg.debug(f"CP       : {child_parameters}")
            lg.debug(f"W        : {self.problem.W}")
            lg.debug(f"DEL ELEMS: {(child != old_child).sum()}")

            # ADD PAHSE
            # trying to add low Values objects if possible
            # i = 0
            # while all(child_parameters <= self.problem.W):
            #     old_child = child.copy()
            #
            #     # add item to solution
            #     child[sorted_value_objects_indexes[i]] = 1
            #
            #     # update parameters
            #     child_parameters = self.problem.df.loc[:, self.problem.df.
            #                                            columns != 'Value'].loc[
            #                                                child ==
            #                                                1].sum().to_numpy()
            #     if not all(child_parameters <= self.problem.W):
            #         child = old_child
            #         break  # TODO: provare a togliere
            #
            #     i = i + 1
            #
            # lg.debug("--- ADD  PHASE ---")
            # lg.debug(f"LAST CHILD: {child}")
            # lg.debug(f"   CP: {child_parameters}")
            # lg.debug(f"OLD CHILD : {old_child}")
            # lg.debug(
            #     f"   CP: {self.problem.df.loc[:, self.problem.df.columns != 'Value'].loc[old_child == 1].sum().to_numpy()}"
            # )
            #
            # lg.debug(f"W         : {self.problem.W}")
            # lg.debug(f"NEW = OLD ?: {True if i != 0 else False}")

            # forse va sempre fatto 🤔
            # if (i - 1) != 0:  # TODO: ricontrollare il -1
            #     child = old_child

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

            lg.debug("population:")
            for elem in self.population:
                lg.debug(f"- {list(elem)}")
            lg.debug(f"fitnesses: {self.f_obj}")

            self.update_best(self.population[0], self.f_obj[0], gen)

        lg.debug("NEW POPULATION: Running")
        select_best()
