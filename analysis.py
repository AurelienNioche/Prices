import json
from pylab import plt, np


def main():

    deviation_from_equilibrium = dict()
    for i in [(1, 0), (2, 0), (2, 1)]:

        deviation_from_equilibrium[i] = json.load(open("../deviation_from_equilibrium_{}.json".format(i), mode="r"))
    x = np.arange(len(list(deviation_from_equilibrium.values())[0]))

    for i in deviation_from_equilibrium.keys():
        plt.plot(x, deviation_from_equilibrium[i])
    plt.show()


if __name__ == "__main__":

    main()