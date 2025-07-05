from GeneticPurchase import GeneticSolver, BruteForceSolver, QuasiBruteForceSolver, GreedyCheapest, GreedyNearest, Graph
import pandas as pd
from tqdm import tqdm
import warnings

import ast
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def main(solver="ga", top_n=None):

    problemset = pd.read_excel("raw_problemset.xlsx")

    if top_n != None: problemset = problemset.iloc[:top_n, :]

    results = []

    l = problemset.shape[0]
    with tqdm(total=l) as pbar:
        for _, problem in problemset.iterrows():

            graph = Graph(graph_filepath=problem["catalog"])
            fuel_price = problem["fuel_price"]
            query = ast.literal_eval(problem["query"])

            if solver=="ga": _solver = GeneticSolver(graph, query, catalog_filepath=problem["catalog"], fuel_cost_per_distance=fuel_price)
            elif solver=="qbf": _solver = QuasiBruteForceSolver(graph, query, catalog_filepath=problem["catalog"], fuel_cost_per_distance=fuel_price)
            elif solver=="gn": _solver = GreedyNearest(graph, query, catalog_filepath=problem["catalog"], fuel_cost_per_distance=fuel_price)
            elif solver=="gc": _solver = GreedyCheapest(graph, query, catalog_filepath=problem["catalog"], fuel_cost_per_distance=fuel_price)
            else: _solver = BruteForceSolver(graph, query, catalog_filepath=problem["catalog"],fuel_cost_per_distance=fuel_price)
            
            result = _solver.solve()
            results.append((_solver.prettify_gene(result), _solver.evaluate_solution(result)))
            pbar.update(1)

    df_result = pd.concat([
        problemset,
        pd.DataFrame(results, columns=[f"{solver} best solution", f"{solver} best value"])
    ], axis=1)
    df_result.to_parquet(f"results_{solver}.parquet")

if __name__ == '__main__':
    main(solver="qbf", top_n=99)
    