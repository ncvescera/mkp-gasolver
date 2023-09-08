import pandas as pd
import argparse
from mkpsolver.problem_representation import MKProblem
from mkpsolver.solver import GeneticAlgorithm
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt

TUNING_SET = [
    "MKP01.txt",
    "MKP03.txt",
    "MKP05.txt",
    "MKP07.txt",
    "MKP11.txt",
    "MKP17.txt",
    "MKP21.txt",
    "MKP23.txt",
    "MKP33.txt",
    # "MKP39.txt",
    "MKP41.txt",
    "MKP47.txt",
]

DATA_FOLDER = "data"


def worker(pcross, pmut, plen, ngen, tk, test_file):
    problem = MKProblem.from_file(test_file)

    solver = GeneticAlgorithm(problem,
                              pcross=pcross,
                              pmut=pmut,
                              num_elem=plen,
                              num_gen=ngen,
                              tk=tk)
    solution = solver.solve()["improvements"][-1]
    return {
        "test_file": test_file,
        "gen": solution[0],
        "found_sol": solution[1],
        "real_sol": problem.sol2,
        "success": solution[1] == problem.sol2,
        "diff": problem.sol2 - solution[1]
    }


def tuning(pmut=.01, pcross=.9, ngen=250, plen=70, tk=45):
    # pmut = .01  # .03, .05
    # pcross = .9  # .97,
    # ngen = 250  # 250,
    # plen = 70  # 57,,,100
    # tk = 45  # 25, , 15, 61

    result = []

    for test in TUNING_SET:
        test_file = f"{DATA_FOLDER}/{test}"

        print("STARTING: ", test_file)

        # creating multiprocess pool and run
        # 5 test in parallel
        p = Pool(processes=5)

        # define worker parameters
        f = partial(worker, pcross, pmut, plen, ngen, tk)

        # starting multiprocessing computation
        p_result = p.map(f, [test_file] * 5)

        for res in p_result:
            result.append(res)

    # results to csv
    df = pd.DataFrame()

    for res in result:
        tmp_df = pd.DataFrame(res.values()).T
        df = pd.concat([df, tmp_df])

    df.columns = ["file", "gen", "found", "target", "success", "diff"]
    df.reset_index(inplace=True, drop=True)
    df.to_csv(
        f"tuning_results_{int(pmut*100)}_{int(pcross*100)}_{ngen}_{plen}_{tk}.csv"
    )

    print(df)


def show_result(path: str):
    path = path.strip()
    df = pd.read_csv(path, index_col=0)

    print(df)
    print()

    fail, success = df['success'].value_counts()
    all_success_ratio = success / (success + fail)
    all_max_diff = max(df['diff'])
    all_min_diff = min(df[df['diff'] > 0]['diff'])

    print(f"SUCCESS (#): {success}, "
          f"FAIL (#): {fail}, "
          f"SUCCESS RATIO (%): {round(all_success_ratio * 100, 1)}, "
          f"MIN DIFF: {all_min_diff}, "
          f"MAX DIFF: {all_max_diff}")
    print()

    df_groupby = df.groupby("file")
    perfile_success = df.groupby(["file"])["success"].value_counts()
    perfile_success = pd.DataFrame(perfile_success)
    perfile_success["%"] = perfile_success['count'] / 5

    # getting SUCCESS # and FAILS # per file
    test = df[['success']]
    test.index = df['file']
    true_list = []
    false_list = []
    for f in test.index.unique():
        f_ture_count = test.loc[f][test.loc[f]['success'] ==
                                   True].count().iloc[0]
        f_false_count = test.loc[f][test.loc[f]['success'] ==
                                    False].count().iloc[0]
        true_list.append(f_ture_count)
        false_list.append(f_false_count)

    perfile_success = pd.DataFrame({"true": true_list, "false": false_list})
    perfile_success.index = test.index.unique()

    # min/max found values
    perfile_minfound = df_groupby["found"].min()
    perfile_maxfound = df_groupby["found"].max()

    perfile_minmaxfound = pd.concat([perfile_minfound, perfile_maxfound],
                                    axis=1)
    perfile_minmaxfound.columns = ["min found", "max found"]

    # min/max/mean diff
    perfile_mindiff = df_groupby["diff"].min()
    perfile_maxdiff = df_groupby["diff"].max()
    perfile_avgdiff = df_groupby["diff"].mean()

    percentage_avgdiff = round(
        (
            perfile_avgdiff /
            # unique return a df with lists of 1 elem
            # with apply I remove the list and get only the element inside
            df_groupby['target'].unique().apply(lambda col: col[0])) * 100,
        1)

    perfile_minmaxdiff = pd.concat(
        [
            perfile_mindiff,
            perfile_maxdiff,
            perfile_avgdiff,
            percentage_avgdiff,
            perfile_success,
        ],
        axis=1,
    )

    perfile_minmaxdiff.columns = [
        "min diff", "max diff", "avg diff", "% avg diff", "success", "fails"
    ]

    perfile_minmax = pd.concat([perfile_minmaxfound, perfile_minmaxdiff],
                               axis=1)

    print("PER FILE STATISTICS")
    print(perfile_minmax)

    # PLOTS
    # perfile_minmax['% avg diff'].plot(kind='bar')
    # plt.savefig(f"{path.split('.')[0]}_percentage_avg_diff.png")
    #
    # perfile_minmax[['success', 'fails']].plot(kind='bar')
    # plt.savefig(f"{path.split('.')[0]}_success_vs_fails.png")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Plot the first chart in the top subplot
    perfile_minmax['% avg diff'].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Percentage AVG diff')
    axes[0].set_ylabel('Percentage')
    axes[0].legend()

    # Plot the second chart in the bottom subplot
    perfile_minmax[['success', 'fails']].plot(kind='bar', ax=axes[1])
    axes[1].set_title(
        f'Success vs Fails\n(ratio: {round(all_success_ratio * 100, 1)}%)')
    axes[1].set_ylabel('Counts')
    axes[1].legend()

    plt.xticks(rotation=45)

    param = path.split('.')[0].split('_')[2:]
    fig.suptitle(
        f'pmut: {float(param[0])/100} pcross: {float(param[1])/100} ngen: {param[2]} plen: {param[3]} tk: {param[4]}',
        fontsize=16)

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plots
    plt.savefig(f"{path.split('.')[0]}.png")


def main(args):
    match args.action:
        case "tuning":
            tuning(ngen=args.number_generation,
                   plen=args.population_lenght,
                   pcross=args.crossover_probability,
                   pmut=args.mutation_probability,
                   tk=args.tournament_k)
        case "plot":
            if args.input is None:
                print("Missing Input file !")
                return

            show_result(args.input)
        case _:
            print("Wrong Action selected !")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter Tuning Script')

    parser.add_argument(
        'action',
        type=str,
        choices=["tuning", "plot"],
        help=
        'Action to run: Tuning Parameters (tuning) or Plotting Results (plot)')
    parser.add_argument('-plen',
                        '--population_lenght',
                        default=16,
                        type=int,
                        help='Initial Population Lenght')
    parser.add_argument('-pcross',
                        '--crossover_probability',
                        default=.9,
                        type=float,
                        help='Crossover Probability (from 0 to 1)')
    parser.add_argument('-pmut',
                        '--mutation_probability',
                        default=.01,
                        type=float,
                        help='Mutation probability (from 0 to 1)')
    parser.add_argument('-ngen',
                        '--number_generation',
                        default=100,
                        type=int,
                        help='Number of generations')
    parser.add_argument('-tk',
                        '--tournament_k',
                        default=5,
                        type=int,
                        help='Tournament random solution to select')
    parser.add_argument('-i',
                        '--input',
                        default=None,
                        type=str,
                        help='Path to result file. Needed if action=plot')

    args = parser.parse_args()
    main(args)
