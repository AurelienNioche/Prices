###############################
#  Based on Gintis, 2007
###############################
import numpy as np
from tqdm import tqdm
from itertools import product
from os import path, mkdir
import json
from scarf_eco import Economy, Agent

debug = False


class RLAgent(Agent):

    def __init__(self, prod, cons, third, prices, production_quantities, idx):

        Agent.__init__(self)
        self.strategies = np.asarray([i for i in product([1, 0, -1], repeat=3)])
        self.strategies_values = np.ones(len(self.strategies)) * 0.5
        self.n_strategies = len(self.strategies)
        self.followed_strategy = 0

        self.change_in_price = 0.01

        self.alpha = 0.5

    def learn(self, reward):

        self.strategies_values[self.followed_strategy] += \
            self.alpha * (reward - self.strategies_values[self.followed_strategy])

    def select_strategy(self):

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        p_values = softmax(self.strategies_values)

        r = np.random.random()

        possible_p_values = list(np.unique(p_values))

        while True:
            if len(possible_p_values) > 1 and r > 0:
                min_in_list = np.min(possible_p_values)
                selected_idx_in_array = np.random.choice(np.where(p_values == min_in_list)[0])

                if min_in_list > r:
                    self.followed_strategy = selected_idx_in_array
                    break
                else:
                    possible_p_values.remove(min_in_list)
                    r -= min_in_list
            else:
                self.followed_strategy = \
                    np.random.choice(np.where(p_values == np.max(np.asarray(possible_p_values)))[0])
                break

    def update_prices(self):

        self.prices[:] = \
            self.prices[:] + self.prices[:] * self.strategies[self.followed_strategy] * self.change_in_price


class RLEconomy(Economy):

    def __init__(self, n_agent, n_generation, n_period_per_generation):

        Economy.__init__(self)  # Will take args by its own (is Cython black magic?)

    def create_agents(self):

        idx = 0

        for i, j, k in [(0, 1, 2),
                        (1, 2, 0),
                        (2, 0, 1)]:

            for n in range(int(self.n_agent / 3)):
                a = RLAgent(
                    prod=i, cons=j, third=k,
                    prices=np.random.random(3),
                    production_quantities=self.production_quantities,
                    idx=idx)

                self.agents.append(a)
                idx += 1

    def play(self):

        # if self.debug:
        #
        #     print("PLAY.")

        self.create_agents()

        # for i in range(self.n_generation):
        for i in tqdm(range(self.n_generation)):

            for j in range(self.n_period_per_generation):

                # if self.debug:
                #
                #     print("")
                #     print("t", i*self.n_period_per_generation + j)

                self.run()

                self.update_agent_fitness()

            self.compute_deviation_from_equilibrium(i)
            self.update_agent_strategies()

        return self.deviation_from_equilibrium

    def update_agent_strategies(self):

        for i in range(self.n_agent):

            self.agents[i].learn(self.fitness[i])
            self.agents[i].select_strategy()
            self.agents[i].update_prices()

        self.fitness[:] = 0


def launch(**kwargs):

    e = RLEconomy(**kwargs)
    return e.play()
