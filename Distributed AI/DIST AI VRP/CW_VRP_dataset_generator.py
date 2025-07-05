from concurrent.futures import ProcessPoolExecutor
from CW_VRP import ClarkWrightSolver, BruteForceSolver, ACOSolver
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 

def main(solver="cw", top_n=None):

    problemset = pd.read_excel("raw_problemset.xlsx")

    if top_n != None: problemset = problemset.iloc[:top_n, :]

    nodes = [pd.read_parquet(i) for i in problemset["nodes"]]
    dms = [pd.read_parquet(i) for i in problemset["distance_matrix"]]
    caps = [i for i in problemset["vehicle_capacity"]]

    if solver=="cw": _solver = ClarkWrightSolver()
    elif solver =="aco": _solver = ACOSolver()
    else: _solver = BruteForceSolver()

    results = []

    l = problemset.shape[0]
    with tqdm(total=l) as pbar:
        with ProcessPoolExecutor(max_workers=5) as executor:
            for result in executor.map(_solver.solve, nodes, dms, caps):
                results.append(result)
                pbar.update(1)
    df_result = pd.concat([
        problemset,
        pd.DataFrame(results, columns=[f"{solver} best route", f"{solver} best value", f"{solver} n vehicles"])
    ], axis=1)
    df_result.to_parquet(f"results_{solver}.parquet")

if __name__ == '__main__':
    main(solver="bf", top_n=19)
    