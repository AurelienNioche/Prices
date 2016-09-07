from os import path, mkdir
import json


def save(results, results_folder="../data"):

    if not path.exists(results_folder):

        mkdir(results_folder)

    for i in results.keys():

        json.dump(results[i].tolist(),
                  open("{}/deviation_from_equilibrium_{}.json".format(results_folder, i), mode="w"), indent=4)
