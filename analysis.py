import json
from pylab import plt, np
from os import path, mkdir


def main(data_folder="../data"):

    deviation_from_equilibrium = dict()
    for i in [(1, 0), (2, 0), (2, 1)]:

        deviation_from_equilibrium[i] = \
            json.load(open("{}/deviation_from_equilibrium_{}.json".format(data_folder, i), mode="r"))

    x = np.arange(len(list(deviation_from_equilibrium.values())[0]))

    fig, ax = plt.subplots()

    for i in deviation_from_equilibrium.keys():
        ax.plot(x, deviation_from_equilibrium[i], label='{} against {}'.format(i[0], i[1]))

    ax.legend(fontsize=12)  # loc='upper center
    ax.set_xlabel("period")
    ax.set_ylabel("actual price / equilibrium price")

    ax.set_title("Price Dynamics in Scarf Three-good Economy \n(relative deviation from equilibrium prices)")

    figure_folder = "../figures"
    if not path.exists(figure_folder):
        mkdir(figure_folder)
    plt.savefig("{}/figure.pdf".format(figure_folder))

    plt.show()


if __name__ == "__main__":

    main(data_folder="../data")