###############################
#  Based on Gintis, 2007
###############################
import numpy as np
import json
from tqdm import tqdm

debug = False


class Agent(object):

    def __init__(self, prod, cons, third, prices, idx):

        self.P = prod
        self.C = cons
        self.T = third
        self.prices = prices

        self.in_hand = {"x": 0., "y": 0., "z": 0.}

        self.exchange = None

        self.utility = 0

        self.idx = idx

        # ---- #
        self.debug = debug

    def are_you_satisfied(self, exchange):

        if self.debug:

            print("Agent", self.idx, "have the proposition ", exchange)
            print("Agent", self.idx, "prices are", self.prices)
            print("Agent", self.idx, "in hand are", self.in_hand)

        # Quantity of good that have been demanded should be > 0
        b_quantity_in_hand = self.in_hand[exchange["b"]["good"]]

        if b_quantity_in_hand > 0:

            # print("Agent", self.idx, "have", b_quantity_in_hand )

            # If agent agrees to prices
            if self.prices[exchange["b"]["good"]] * exchange["b"]["quantity"] < \
                    self.prices[exchange["a"]["good"]] * exchange["b"]["quantity"]:

                # If agent have enough b in hand
                if self.in_hand[exchange["b"]["good"]] >= exchange["b"]["quantity"]:

                    self.in_hand[exchange["b"]["good"]] -= exchange["b"]["quantity"]
                    self.in_hand[exchange["a"]["good"]] += exchange["a"]["quantity"]

                    return "Agree", exchange

                else:

                    b_quantity = self.in_hand[exchange["b"]["good"]]
                    self.in_hand[exchange["b"]["good"]] = 0
                    a_quantity = (b_quantity/exchange["b"]["quantity"]) * exchange["a"]["quantity"]

                    output = exchange.copy()
                    output["b"]["quantity"] = b_quantity
                    output["a"]["quantity"] = a_quantity

                    return "Agree", output

            else:

                return "PricesToHigh", None

        else:

            return "NotPossessed", None

    # Based on p. 1286
    def meet(self, other_agent, total_number_of_objects):

        if self.debug:

            print("Agent", self.idx, "propose an exchange to agent", other_agent.idx, ".")
            print("Agent", self.idx, "produce", self.P, "and consume", self.C, ".")
            print("Agent", self.idx, "prices are", self.prices, ".")
            print("Agent", self.idx, "in hand are", self.in_hand, ".")

        self.exchange = None

        # If T in hand
        if self.in_hand[self.T] > 0:

            # print("Agent", self.idx, "has T in hand", self.in_hand[self.T]), "."

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

        if self.exchange is not None:

            if self.debug:

                print("exchange")

            self.proceed_to_exchange()

    def propose_an_exchange(self, object_a, object_b, other_agent, quantity=1.):

        a_quantity = self.in_hand[object_a] * quantity
        b_quantity = (self.in_hand[object_a] * quantity) * (self.prices[object_a] / self.prices[object_b])

        if self.debug:

            print('')
            print("Agent", self.idx, "propose ", object_a, "against", object_b, ".")
            print("Agent", self.idx, "propose", a_quantity, "in quantity against", b_quantity, ".")
            print('')

        other_agent_response = other_agent.are_you_satisfied(
            exchange={
                "a": {"quantity": a_quantity, "good": object_a},
                "b": {"quantity": b_quantity,
                      "good": object_b}
            }
        )
        agree = other_agent_response[0] == "Agree"

        if agree:

            self.exchange = other_agent_response[1]

        return agree

    def proceed_to_exchange(self):

        self.in_hand[self.exchange["a"]["good"]] -= self.exchange["a"]["quantity"]
        self.in_hand[self.exchange["b"]["good"]] += self.exchange["b"]["quantity"]

    def consume(self, total_number_of_objects):

        self.utility += np.min([self.in_hand[self.P]/total_number_of_objects[self.P],
                                self.in_hand[self.C]/total_number_of_objects[self.C]])

        self.in_hand[self.C] = 0
        self.in_hand[self.P] = 0

    def produce(self, production_quantity):

        for i, j in self.in_hand.items():

            self.in_hand[i] = 0

        self.in_hand[self.P] = production_quantity

        if self.debug:
            print("In hand.", self.in_hand)


class Economy(object):

    def __init__(self):

        self.n_agent = 1000 * 3  # original: 1000 * 3
        self.n_generation = 1500  # original: 2500
        self.n_period_per_generation = 10  # original: 10

        self.agents = []
        self.total_number_of_objects = {
            "x": 10,
            "y": 20,
            "z": 400
        }
        self.equilibrium_prices = {
            "x": self.total_number_of_objects["z"]/self.total_number_of_objects["x"],
            "y": self.total_number_of_objects["z"]/self.total_number_of_objects["y"],
            "z": 1.
        }
        self.n_reproduction_pair = int((0.05 * self.n_agent)/2.)  # So 50 taken in a all, which means 0.5% of the population
        # print("n reproduction pair", self.n_reproduction_pair)
        self.p_mutation = 0.1

        # -------- #

        self.types = np.asarray([0, ] * int(self.n_agent/3) +
                                [1, ] * int(self.n_agent/3) +
                                [2, ] * int(self.n_agent/3))

        self.idx = np.asarray([np.where(self.types == 0)[0], np.where(self.types == 1)[0], np.where(self.types == 2)[0]])

        self.fitnesses_payoffs = np.zeros(self.n_agent)

        self.deviation_from_equilibrium = {}
        for i in [("z", "x"), ("z", "y"), ("y", "x")]:

            self.deviation_from_equilibrium[i] = []

        # ----- #

        self.debug = debug

    def play(self):

        if self.debug:

            print("PLAY.")

        self.create_agents()

        #for i in range(self.n_generation):
        for i in tqdm(range(self.n_generation)):

            for j in range(self.n_period_per_generation):

                if self.debug:

                    print("")
                    print("t", i*self.n_period_per_generation + j)

                self.run()
                self.compute_deviation_from_equilibrium()
                self.update_agent_fitnesses_payoff()

            self.reproduce_agents()

        self.save()

    def update_agent_fitnesses_payoff(self):

        if self.debug:
            print("Update agent fitness.")

        for i in range(self.n_agent):

            self.fitnesses_payoffs[i] += self.agents[i].utility
            self.agents[i].utility = 0

    def reproduce_agents(self):

        if self.debug:
            print("")
            print("Reproduce.")

        idx_to_reproduce = np.random.permutation(self.n_agent)[:self.n_reproduction_pair]

        # if self.debug:
        #     print("idx_to_reproduce", idx_to_reproduce)

        for i in idx_to_reproduce:

            i_type = self.types[i]
            other_agent = i
            while other_agent == i:
                other_agent = np.random.choice(self.idx[i_type])

            if self.debug:
                print("i", i)
                print('other_agent', other_agent)

                print("fitness other_agent", self.fitnesses_payoffs[other_agent])
                print("fitness i", self.fitnesses_payoffs[i])

            if self.fitnesses_payoffs[other_agent] > self.fitnesses_payoffs[i]:

                to_be_copied = other_agent
                to_be_changed = i

            elif self.fitnesses_payoffs[other_agent] < self.fitnesses_payoffs[i]:

                to_be_copied = i
                to_be_changed = other_agent

            else:
                continue

            # No specific rule if agents' fitness is the same
            if self.debug:
                print('to be copied', to_be_copied)
                print("to be changed", to_be_changed)

            for key, value in self.agents[to_be_copied].prices.items():

                r = np.random.random()
                if r > self.p_mutation:

                    self.agents[to_be_changed].prices[key] = value
                    if self.debug:
                        print("no mutated", key, value)
                else:
                    self.agents[to_be_changed].prices[key] = value + 0.10 * value * np.random.choice([1., -1.])

                    if self.debug:
                        print("mutated", key, value)
                        print("prices", self.agents[to_be_changed].prices[key])

        # Reinitialize fitness counter
        self.fitnesses_payoffs[:] = 0

    def run(self):

        if self.debug:
            print("Run one round.")

        # Each agent produces at the beginning of each round. He gets free of other goods he could have.
        for i in range(self.n_agent):

            self.agents[i].produce(production_quantity=self.total_number_of_objects[self.agents[i].P])

        # Take a random order among the indexes of the agents.
        random_order = np.random.permutation(self.n_agent)

        for i in random_order:

            # Each agent is "initiator' of an exchange during one period.
            initiator = self.agents[i]

            # A 'responder' is randomly selected.
            responder = self.agents[np.random.choice(np.arange(self.n_agent))]

            # The 'initiator' can make a series of propositions to the responder.
            initiator.meet(other_agent=responder, total_number_of_objects=self.total_number_of_objects)

        # Each agent consumes at the end of each round.
        for i in range(self.n_agent):

            self.agents[i].consume(total_number_of_objects=self.total_number_of_objects)

    def create_agents(self):

        idx = 0

        for i, j, k in [("x", "y", "z"),
                        ("y", "z", "x"),
                        ("z", "x", "y")]:

            for n in range(int(self.n_agent/3)):

                a = Agent(prod=i, cons=j, third=k,
                          prices={"x": np.random.random(),
                                  "y": np.random.random(),
                                  "z": np.random.random()},
                          idx=idx)

                self.agents.append(a)
                idx += 1

    def compute_deviation_from_equilibrium(self, debug=False):

        if debug:

            print("Compute deviation from equilibrium.")

        for a, b in [("z", "x"), ("z", "y"), ("y", "x")]:

            prices_for_a = []
            prices_for_b = []
            for i in range(self.n_agent):

                prices_for_a.append(self.agents[i].prices[a])
                prices_for_b.append(self.agents[i].prices[b])

            deviation_a_against_b = \
                (np.mean(prices_for_a)/np.mean(prices_for_b)) / \
                (self.equilibrium_prices[a]/self.equilibrium_prices[b])

            self.deviation_from_equilibrium[(a, b)].append(
                deviation_a_against_b

            )

            if debug:

                print(a, ":", np.mean(prices_for_a))
                print(b, ":", np.mean(prices_for_b))
                print("")

                print(a, "eq:", self.equilibrium_prices[a])
                print(b, "eq:", self.equilibrium_prices[b])
                print('')

                print("deviation", a, b, ":", deviation_a_against_b)

    def save(self):

        for i in self.deviation_from_equilibrium.keys():

            json.dump(self.deviation_from_equilibrium[i],
                      open("../deviation_from_equilibrium_{}.json".format(i), mode="w"), indent=4)

    def __del__(self):

        self.save()


def launch():

    e = Economy()
    e.play()

if __name__ == "__main__":

    launch()