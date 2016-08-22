# from os import system
# system("python3 make_file.py")
###############################
###############################
from scarf_eco import launch
from os import path, mkdir
import json


def save(results, results_folder="../data"):

    if not path.exists(results_folder):

        mkdir(results_folder)

    for i in results.keys():

        json.dump(results[i].tolist(),
                  open("{}/deviation_from_equilibrium_{}.json".format(results_folder, i), mode="w"), indent=4)


def main():

    results = launch(
        n_agent=1000 * 3,  # original: 1000 * 3
        n_generation=1000,  # original: 2500
        n_period_per_generation=10  # original: 10
    )
    save(results)


if __name__ == "__main__":

    main()
