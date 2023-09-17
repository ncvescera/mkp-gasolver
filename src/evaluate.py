import pandas as pd
import argparse
from mkpsolver.problem_representation import MKProblem
from mkpsolver.solver import GeneticAlgorithm
from multiprocessing import Pool
from functools import partial
from tuning import show_result

TESTING_SET = [
    "MKP02.txt",
    "MKP04.txt",
    "MKP06.txt",
    "MKP08.txt",
    "MKP10.txt",
    "MKP12.txt",
    "MKP20.txt",
    "MKP22.txt",
    "MKP24.txt",
    "MKP30.txt",
    "MKP36.txt",
    "MKP40.txt",
    "MKP46.txt",
    "MKP48.txt",
    "MKP50.txt",
    "MKP54.txt",
]

TESTING_PARAMETER = [
    {
        "pmut": .05,
        "pcross": .99,
        "ngen": 250,
        "plen": 100,
        "tk": 80
    },
    {
        "pmut": .02,
        "pcross": .97,
        "ngen": 250,
        "plen": 100,
        "tk": 40
    },
    {
        "pmut": .05,
        "pcross": .97,
        "ngen": 250,
        "plen": 100,
        "tk": 61
    },
]

DATA_FOLDER = "data"


def worker(pcross, pmut, plen, ngen, tk, test_file):
    problem = MKProblem.from_file(test_file)

    solver = GeneticAlgorithm(
        problem,
        pcross=pcross,
        pmut=pmut,
        num_elem=plen,
        num_gen=ngen,
        tk=tk,
    )
    solution = solver.solve()["improvements"][-1]

    return {
        "test_file": test_file,
        "gen": solution[0],
        "found_sol": solution[1],
        "real_sol": problem.sol2,
        "success": solution[1] == problem.sol2,
        "diff": problem.sol2 - solution[1]
    }


def evaluate(pmut=.01, pcross=.9, ngen=250, plen=70, tk=45):
    result = []

    for test in TESTING_SET:
        test_file = f"{DATA_FOLDER}/{test}"

        print("STARTING: ", test_file)

        # creating multiprocess pool and run
        # 5 test in parallel
        p = Pool(processes=10)

        # define worker parameters
        f = partial(worker, pcross, pmut, plen, ngen, tk)

        # starting multiprocessing computation
        p_result = p.map(f, [test_file] * 10)

        for res in p_result:
            result.append(res)

    # results to csv
    df = pd.DataFrame()

    for res in result:
        tmp_df = pd.DataFrame(res.values()).T
        df = pd.concat([df, tmp_df])

    df.columns = ["file", "gen", "found", "target", "success", "diff"]
    df.reset_index(inplace=True, drop=True)
    df = df.round(1)
    df.to_csv(
        f"evaluation_results_{int(PMUT*100)}_{int(PCROSS*100)}_{NGEN}_{PLEN}_{TK}.csv"
    )

    print(df)


def main(args):
    match args.action:
        case "evaluate":
            for param in TESTING_PARAMETER:
                evaluate(ngen=param['ngen'],
                         plen=param['plen'],
                         pcross=param['pcross'],
                         pmut=param['pmut'],
                         tk=param['tk'])

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
        choices=["evaluate", "plot"],
        help=
        'Action to run: Evaluate Parameters (evaluate) or Plotting Results (plot)',
    )
    parser.add_argument('-i',
                        '--input',
                        default=None,
                        type=str,
                        help='Path to result file. Needed if action=plot')

    args = parser.parse_args()
    main(args)
