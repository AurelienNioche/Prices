###############################
#  Based on Gintis, 2007
###############################
import numpy as np
cimport numpy as cnp
import json
from tqdm import tqdm
from cpython cimport bool

debug = False


cdef class Agent(object):

    cdef:
        public:
            int P, C, T, idx
            cnp.ndarray in_hand, prices, production_quantities
            float utility


    def __cinit__(self, int prod, int cons, int third, cnp.ndarray prices,
                  cnp.ndarray production_quantities, int idx):

        self.P = prod
        self.C = cons
        self.T = third
        self.prices = prices
        self.production_quantities = production_quantities

        self.in_hand = np.zeros(3)
        self.utility = 0

        self.idx = idx

    cpdef cnp.ndarray are_you_satisfied(self, int object_a, int object_b, float a_quantity, float b_quantity):

        cdef:
            float b_quantity_in_hand, new_a_quantity

        # if self.debug:
        #
        #     print("Agent", self.idx, "have the proposition ", exchange)
        #     print("Agent", self.idx, "prices are", self.prices)
        #     print("Agent", self.idx, "in hand are", self.in_hand)

        # Quantity of good that have been demanded should be > 0
        b_quantity_in_hand = self.in_hand[object_b]

        if b_quantity_in_hand > 0:

            # print("Agent", self.idx, "have", b_quantity_in_hand )

            # If agent agrees to prices
            if self.prices[object_b] * b_quantity < \
                    self.prices[object_a] * a_quantity:

                # If agent have enough b in hand
                if self.in_hand[object_b] >= b_quantity:

                    self.in_hand[object_b] -= b_quantity
                    self.in_hand[object_a] += a_quantity
                    
                    self.consume()

                    return np.array([a_quantity, b_quantity])

                else:

                    self.in_hand[object_b] = 0
                    new_a_quantity = (b_quantity_in_hand/b_quantity) * a_quantity

                    return np.array([new_a_quantity, b_quantity_in_hand])

            else:

                return np.zeros(1)

        else:

            return np.zeros(1)

    # Based on p. 1286
    cpdef meet(self, object other_agent):

        cdef:

            bool other_agent_agreeing

        # if self.debug:
        #
        #     print("Agent", self.idx, "propose an exchange to agent", other_agent.idx, ".")
        #     print("Agent", self.idx, "produce", self.P, "and consume", self.C, ".")
        #     print("Agent", self.idx, "prices are", self.prices, ".")
        #     print("Agent", self.idx, "in hand are", self.in_hand, ".")

        # If T in hand
        if self.in_hand[self.T] > 0:

            # Begin by proposing T against C
            other_agent_agreeing = self.propose_an_exchange(self.T, self.C, other_agent)

            if not other_agent_agreeing:

                # ... propose T against P
                self.propose_an_exchange(self.T, self.P, other_agent)

        # If P in hand
        elif self.in_hand[self.P] > 0:

            # Begin by proposing P against C:
            other_agent_agreeing = self.propose_an_exchange(self.P, self.C, other_agent)

            if not other_agent_agreeing:

                # ... propose half-P against T
                self.propose_an_exchange(self.P, self.T, other_agent, quantity=0.5)

        # If neither P or T in hand, but only C:
        elif self.in_hand[self.C] > 0:

            # Begin by proposing C against P
            other_agent_agreeing = self.propose_an_exchange(self.C, self.P, other_agent)

            if not other_agent_agreeing:

                # ... propose C against T
                self.propose_an_exchange(self.C, self.T, other_agent)

    cpdef bool propose_an_exchange(self, int object_a, int object_b, object other_agent, float quantity=1.):

        cdef:
            float a_quantity
            float b_quantity
            cnp.ndarray other_agent_response
            bool agree
            cnp.ndarray goods_to_exchange

        a_quantity = self.in_hand[object_a] * quantity
        b_quantity = (self.in_hand[object_a] * quantity) * (self.prices[object_a] / self.prices[object_b])

        # if self.debug:
        #
        #     print('')
        #     print("Agent", self.idx, "propose ", object_a, "against", object_b, ".")
        #     print("Agent", self.idx, "propose", a_quantity, "in quantity against", b_quantity, ".")
        #     print('')

        other_agent_response = other_agent.are_you_satisfied(
            object_a, object_b, a_quantity, b_quantity
        )

        exchange_occurs = len(other_agent_response) > 1

        if exchange_occurs:

            self.proceed_to_exchange(object_a, object_b, other_agent_response[0], other_agent_response[1])
            self.consume()

        return exchange_occurs

    cdef proceed_to_exchange(self, int object_a, int object_b, float a_quantity, float b_quantity):

        self.in_hand[object_a] -= a_quantity
        self.in_hand[object_b] += b_quantity

    cpdef consume(self):

        self.utility += np.min([self.in_hand[self.P]/self.production_quantities[self.P],
                                self.in_hand[self.C]/self.production_quantities[self.C]])

        self.in_hand[self.C] = 0
        self.in_hand[self.P] = 0

    cpdef produce(self):

        self.in_hand[:] = 0
        self.in_hand[self.P] = self.production_quantities[self.P]

        # if self.debug:
        #     print("In hand.", self.in_hand)


cdef class Economy(object):

    cdef:
        int n_agent, n_generation, n_period_per_generation, n_reproduction_pair, t
        list agents
        dict deviation_from_equilibrium
        float reproduction_rate, p_mutation
        cnp.ndarray types, idx, fitness, equilibrium_prices, production_quantities
        bool debug

    def __cinit__(self, int n_agent, int n_generation, int n_period_per_generation):

        self.n_agent = n_agent  # original: 1000 * 3
        self.n_generation = n_generation  # original: 2500
        self.n_period_per_generation = n_period_per_generation  # original: 10

        self.agents = []
        self.production_quantities = np.array([10, 20, 400])

        self.equilibrium_prices = np.array([
            self.production_quantities[2]/self.production_quantities[0],
            self.production_quantities[2]/self.production_quantities[1],
            1.
        ])

        self.reproduction_rate = 0.05
        self.n_reproduction_pair = int((self.reproduction_rate * self.n_agent)/2.)
        # print("n reproduction pair", self.n_reproduction_pair)
        self.p_mutation = 0.1

        # -------- #

        self.t = 0

        self.types = np.asarray([0, ] * int(self.n_agent/3) +
                                [1, ] * int(self.n_agent/3) +
                                [2, ] * int(self.n_agent/3))

        self.idx = np.asarray([np.where(self.types == 0)[0],
                               np.where(self.types == 1)[0],
                               np.where(self.types == 2)[0]])

        self.fitness = np.zeros(self.n_agent)

        self.deviation_from_equilibrium = {}
        for i in [(2, 0), (2, 1), (1, 0)]:

            self.deviation_from_equilibrium[i] = np.zeros(self.n_generation*self.n_period_per_generation)

        # ----- #

        self.debug = debug

    cpdef play(self):

        cdef:
            int i, j

        # if self.debug:
        #
        #     print("PLAY.")

        self.create_agents()

        #for i in range(self.n_generation):
        for i in tqdm(range(self.n_generation)):

            for j in range(self.n_period_per_generation):

                self.t = i*self.n_period_per_generation + j

                # if self.debug:
                #
                #     print("")
                #     print("t", i*self.n_period_per_generation + j)

                self.run()
                self.compute_deviation_from_equilibrium()
                self.update_agent_fitness()

            self.reproduce_agents()

        return self.deviation_from_equilibrium

    cdef update_agent_fitness(self):

        cdef:
            i

        # if self.debug:
        #     print("Update agent fitness.")

        for i in range(self.n_agent):

            self.fitness[i] += self.agents[i].utility
            self.agents[i].utility = 0

    cdef reproduce_agents(self):

        cdef:

            cnp.ndarray idx_to_reproduce
            int i, to_be_copied, to_be_changed, price
            cnp.ndarray r

        # if self.debug:
        #     print("")
        #     print("Reproduce.")

        idx_to_reproduce = np.random.permutation(self.n_agent)[:self.n_reproduction_pair]

        # if self.debug:
        #     print("idx_to_reproduce", idx_to_reproduce)

        for i in idx_to_reproduce:

            i_type = self.types[i]
            other_agent = i
            while other_agent == i:
                other_agent = np.random.choice(self.idx[i_type])

            # if self.debug:
            #     print("i", i)
            #     print('other_agent', other_agent)
            #
            #     print("fitness other_agent", self.fitness[other_agent])
            #     print("fitness i", self.fitness[i])

            if self.fitness[other_agent] > self.fitness[i]:

                to_be_copied = other_agent
                to_be_changed = i

            elif self.fitness[other_agent] < self.fitness[i]:

                to_be_copied = i
                to_be_changed = other_agent

            else:
                continue

            # No specific rule if agents' fitness is the same
            # if self.debug:
            #     print('to be copied', to_be_copied)
            #     print("to be changed", to_be_changed)

            r = np.random.random(3)

            for price in range(3):

                value = self.agents[to_be_copied].prices[price]

                if r[price] > self.p_mutation:

                    self.agents[to_be_changed].prices[price] = value
                    # if self.debug:
                    #     print("no mutated", price, value)
                else:
                    self.agents[to_be_changed].prices[price] = value + 0.10 * value * np.random.choice([1., -1.])

                    # if self.debug:
                    #     print("mutated", price, value)
                    #     print("prices", self.agents[to_be_changed].prices[price])

        # Reinitialize fitness counter
        self.fitness[:] = 0

    cdef run(self):

        cdef:
            int i
            cnp.ndarray random_order
            object initiator, responder
            float production_quantity

        # if self.debug:
        #     print("Run one round.")

        # Each agent produces at the beginning of each round. He gets free of other goods he could have.
        for i in range(self.n_agent):

            production_quantity = self.production_quantities[self.agents[i].P]

            self.agents[i].produce()

        # Take a random order among the indexes of the agents.
        random_order = np.random.permutation(self.n_agent)

        for i in random_order:

            # Each agent is "initiator' of an exchange during one period.
            initiator = self.agents[i]

            # A 'responder' is randomly selected.
            responder = self.agents[np.random.choice(np.arange(self.n_agent))]

            # The 'initiator' can make a series of propositions to the responder.
            initiator.meet(responder)

        # # Each agent consumes at the end of each round.
        # for i in range(self.n_agent):
        # 
        #     self.agents[i].consume(self.production_quantities)

    cdef create_agents(self):

        cdef:
            int idx, i, j, k
            object a


        idx = 0

        for i, j, k in [(0, 1, 2),
                        (1, 2, 0),
                        (2, 0, 1)]:

            for n in range(int(self.n_agent/3)):

                a = Agent(prod=i, cons=j, third=k,
                          prices=np.random.random(3),
                          production_quantities=self.production_quantities,
                          idx=idx)

                self.agents.append(a)
                idx += 1

    cdef compute_deviation_from_equilibrium(self):

        cdef:
            int  a, b, i
            cnp.ndarray prices_for_a, prices_for_b
            float deviation_a_against_b

        # if self.debug:
        #
        #     print("Compute deviation from equilibrium.")

        for a, b in [(2, 0), (2, 1), (1, 0)]:

            prices_for_a = np.zeros(self.n_agent)
            prices_for_b = np.zeros(self.n_agent)
            for i in range(self.n_agent):

                prices_for_a[i] = self.agents[i].prices[a]
                prices_for_b[i] = self.agents[i].prices[b]

            deviation_a_against_b = \
                (np.mean(prices_for_a)/np.mean(prices_for_b)) / \
                (self.equilibrium_prices[a]/self.equilibrium_prices[b])

            self.deviation_from_equilibrium[(a, b)][self.t] = deviation_a_against_b

            # if debug:
            #
            #     print(a, ":", np.mean(prices_for_a))
            #     print(b, ":", np.mean(prices_for_b))
            #     print("")
            #
            #     print(a, "eq:", self.equilibrium_prices[a])
            #     print(b, "eq:", self.equilibrium_prices[b])
            #     print('')
            #
            #     print("deviation", a, b, ":", deviation_a_against_b)


def launch(**kwargs):

    e = Economy(**kwargs)
    return e.play()
