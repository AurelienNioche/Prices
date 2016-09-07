from py_scarf_eco_with_RL import launch
from save import save


def main():

    results = launch(
        n_agent=1000 * 3, n_generation=1000, n_period_per_generation=10
    )
    save(save(results=results, results_folder="../dataRL"))

if __name__ == "__main__":

    main()
