import pandas as pd
from mkpsolver.problem_representation import MKProblem
from mkpsolver.solver import GeneticAlgorithm
from tqdm import tqdm

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


def main():
    pmut = .01  # .03, .05
    pcross = .9  # .97,
    ngen = 250  # 250,
    plen = 70  # 57,,,100
    tk = 45  # 25, , 15, 61

    result = []

    for test in TUNING_SET:
        test_file = f"{DATA_FOLDER}/{test}"

        print("STARTING: ", test_file)
        for i in tqdm(range(5)):
            problem = MKProblem.from_file(test_file)

            solver = GeneticAlgorithm(problem,
                                      pcross=pcross,
                                      pmut=pmut,
                                      num_elem=plen,
                                      num_gen=ngen,
                                      tk=tk)
            solution = solver.solve()["improvements"][-1]
            result.append({
                "test_file": test_file,
                "gen": solution[0],
                "found_sol": solution[1],
                "real_sol": problem.sol2,
                "success": solution[1] == problem.sol2,
                "diff": problem.sol2 - solution[1]
            })

    print(result)

    df = pd.DataFrame()

    for res in result:
        tmp_df = pd.DataFrame(res.values()).T
        df = pd.concat([df, tmp_df])

    df.columns = ["file", "gen", "found", "target", "success", "diff"]
    df.reset_index(inplace=True, drop=True)
    df.to_csv(f"tuning_results_{pmut*100}_{pcross*100}_{ngen}_{plen}_{tk}.csv")

    print(df)


if __name__ == "__main__":
    main()
