# from os import system
# system("python3 make_file.py")
###############################
###############################
from scarf_eco import launch
from save import save


def main():

    results = launch(
        n_agent=1000 * 3,  # original: 1000 * 3
        n_generation=20000,  # original: 2500
        n_period_per_generation=10  # original: 10
    )
    save(results=results, results_folder="../data")


if __name__ == "__main__":

    main()
