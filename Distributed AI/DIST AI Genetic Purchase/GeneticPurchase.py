import pandas as pd
import numpy as np
import random
import itertools

class Encoder():

  def __init__(self, values_to_encode=None):
    self.encoding_dict = None
    self.decoding_dict = None
    if values_to_encode: self.build_encodings(values_to_encode)

  def build_encodings(self, values):
    self.encoding_dict = {v: i for i, v in enumerate(values)}
    self.decoding_dict = {i: v for i, v in enumerate(values)}

  def encode(self, value):
    return self.encoding_dict[value]

  def decode(self, value):
    return self.decoding_dict[value]
  
class Graph():

  def __init__(self, graph_filepath=None):
    self.edges = None
    self.place_encoder = None
    self.distance_matrix = None
    if graph_filepath: self.load_graph_from_excel(graph_filepath)

  def load_graph_from_excel(self, filepath, sheet_name="grafo"):
    self.edges = pd.read_excel(filepath, sheet_name=sheet_name)
    unique_places = list(set(
        list(self.edges["edge 1"].values) +
        list(self.edges["edge 2"].values)
    ))
    n_unique_places = len(unique_places)
    self.place_encoder = Encoder(list(unique_places))
    self.distance_matrix = np.full((n_unique_places, n_unique_places), np.inf)
    for _, row in self.edges.iterrows():
      (
        self.distance_matrix
        [self.place_encoder.encode(row["edge 1"])]
        [self.place_encoder.encode(row["edge 2"])]
      ) = row["distance"]
    self.distance_matrix[np.eye(n_unique_places, dtype="bool")] = 0
    self.__reduce_distance_matrix()

  def __reduce_distance_matrix(self):
    n = self.distance_matrix.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                self.distance_matrix[i][j] = min(
                    self.distance_matrix[i][j],
                    self.distance_matrix[i][k] + self.distance_matrix[k][j]
                )

class Solver():
    
  def __init__(self, graph, product_query,
               start_place_name="Casa", fuel_cost_per_distance=.1, catalog_filepath=None):
    self.graph = graph
    self.hub_id = self.graph.place_encoder.encode(start_place_name)
    self.product_query = product_query
    self.fuel_cost = fuel_cost_per_distance
    self.catalog = None
    self.product_encoder = None
    #self.store_encoder = None
    self.price_matrix = None
    self.distance_matrix = None
    if catalog_filepath: self.load_catalog_from_excel(catalog_filepath)

  def load_catalog_from_excel(self, filepath, sheet_name="catalogo"):
    self.catalog = pd.read_excel(filepath, sheet_name=sheet_name)
    self.product_encoder = Encoder(list(self.catalog["product"].unique()))
    #self.store_encoder = Encoder(list(self.catalog["store"].unique()))
    self.price_matrix = (
      self.catalog
      .pivot(index="product", columns="store", values="price")
      .rename(index = self.product_encoder.encode, columns = self.graph.place_encoder.encode)
      .sort_index(axis=0)
      .sort_index(axis=1)
      #.to_numpy()
    )

class GeneticSolver(Solver):

  def __initialize_population(self, size):
    store_ids = list(self.price_matrix.columns)
    product_ids = [self.product_encoder.encode(i) for i in self.product_query]
    return [[(random.choice(store_ids), product) for product in product_ids] for i in range(size)]

  def __get_total_product_cost(self, gene):
    return sum([self.price_matrix[store][product] for store, product in gene])

  def __get_total_travel_cost(self, gene):
    travel_route = list(zip(
        [self.hub_id] + [store for store, _ in gene],
        [store for store, _ in gene] + [self.hub_id]
    ))
    return sum([self.graph.distance_matrix[store_a][store_b] * self.fuel_cost for store_a, store_b in travel_route])

  def evaluate_solution(self, gene):
    return self.__fitness(gene)

  def prettify_gene(self, gene):
    store_ids, product_ids = list(zip(*gene))
    stores = [self.graph.place_encoder.decode(s) for s in store_ids]
    products = [self.product_encoder.decode(p) for p in product_ids]
    return list(zip(stores, products))

  def __fitness(self, gene, replace_nan_with=np.nan):
    fitness_value = self.__get_total_product_cost(gene) + self.__get_total_travel_cost(gene)
    if np.isnan(fitness_value): fitness_value = replace_nan_with
    return fitness_value

  def solve(self, n_iters=1000, pop_size=100, selection_rate=0.5, mutation_prob=0.1):
    population = self.__initialize_population(pop_size)
    selection_size = int(pop_size * selection_rate)
    children_amount = pop_size - selection_size
    for _ in range(n_iters):
      best_individuals = self.__selection(population, selection_size)
      children = self.__crossover(best_individuals, children_amount)
      #children = self.__mutation(children, mutation_prob)
      population = best_individuals + children
    return self.__selection(population, 1)[0]

  def __selection(self, genes, selection_size):
    return sorted(genes, key = lambda x: self.__fitness(x, replace_nan_with=np.inf))[:selection_size]

  def __crossover(self, genes, children_amount):
    children = []
    while len(children) < children_amount:
      parent_1 = random.choice(genes)
      parent_2 = random.choice(genes)
      p1_stores, p1_products = list(zip(*parent_1))
      p2_stores, p2_products = list(zip(*parent_2))
      product_crossover_children = self.__partially_mapped_crossover(p1_products, p2_products)
      store_crossover_children = self.__one_point_crossover(p1_stores, p2_stores)
      child_1 = list(zip(store_crossover_children[0], product_crossover_children[0]))
      child_2 = list(zip(store_crossover_children[1], product_crossover_children[1]))
      children += [child_1, child_2]
    return children[:children_amount]

  def __one_point_crossover(self, parent_1, parent_2, cross_point=None):
    if cross_point is None: cross_point = random.randint(0, len(parent_1)-1)
    child_1 = parent_1[:cross_point] + parent_2[cross_point:]
    child_2 = parent_2[:cross_point] + parent_1[cross_point:]
    return child_1, child_2

  def __partially_mapped_crossover(self, parent_1, parent_2, cross_point=None):
    if cross_point is None: cross_point = random.randint(0, len(parent_1)-1)
    mappings = {a: b for a, b in zip(parent_1[:cross_point], parent_2[:cross_point])}
    mappings.update({b: a for a, b in zip(parent_1[:cross_point], parent_2[:cross_point])})
    child_1 = parent_1[:cross_point] + tuple(mappings.get(i, i) for i in parent_2[cross_point:])
    child_2 = parent_2[:cross_point] + tuple(mappings.get(i, i) for i in parent_1[cross_point:])
    return child_1, child_2, mappings

  def __mutation(self, genes, mutation_prob):
    for gene in genes:
      if random.random() < mutation_prob:
        gene = self.__swap_mutation(gene)
    return genes

  def __swap_mutation(self, gene):
     idx = range(len(gene))
     i1, i2 = random.sample(idx, 2)
     gene[i1], gene[i2] = gene[i2], gene[i1]
     return gene

class GreedyNearest(GeneticSolver):

    """
    Prioritizes supermarkets closer to the user
    """

    def solve(self):

        nearest_place = self.graph.place_encoder.encoding_dict["Casa"]
        product_ids = [self.product_encoder.encode(i) for i in self.product_query]

        # Preprocess distance matrix
        tdm = self.graph.distance_matrix.copy()
        place_ids = list(self.graph.place_encoder.encoding_dict.values())
        store_ids = list(self.price_matrix.columns)
        for p in place_ids:
            if p not in store_ids:
                tdm[:, p] = np.inf
                
        solution = []
        explored_products = set()

        while len(solution) != len(self.product_query):
            tdm[:, nearest_place] = np.inf
            nearest_place = np.argmin(tdm[nearest_place])
            products = self.price_matrix.loc[:, nearest_place]
            products = [i for i in products[products.notnull()].index if i not in explored_products and i in product_ids]
            explored_products.update(products)
            solution += zip([nearest_place] * len(products), products)

        return solution
    
class GreedyCheapest(GeneticSolver):

    """
    Prioritizes cheapest products for the user
    """

    def solve(self):

        product_ids = [self.product_encoder.encode(i) for i in self.product_query]
        solution = []
        for i, j in self.price_matrix.loc[product_ids, :].iterrows():
            solution.append((self.price_matrix.columns[j.argmin()], i))
        solution.sort(key = lambda x : x[0])

        return solution
    
class BruteForceSolver(GeneticSolver):

    """
    This solver searches all possible solutions
    """

    def _backtrack(self, combination, index):
        
        if index == len(self.product_query):
            return [combination[:]]

        results = []
        for store in self.price_matrix.columns:
            product = self.price_matrix.index[index]
            if not pd.isna(self.price_matrix.loc[product, store]):
                combination.append((store, product))
                results += self._backtrack(combination, index + 1)
                combination.pop()
        
        return results
    
    def _evaluate_bf_solutions(self, store_combinations, catalog_combinations):

        best_val = np.inf
        best_comb = None

        for place_order in store_combinations:
            for comb in catalog_combinations:
                # Reorder comb
                new_comb = []
                for order in place_order:
                    new_comb += [i for i in comb if i[0] == order]
                # Evaluate and update
                val = self.evaluate_solution(new_comb)
                if val != np.nan and val < best_val:
                    best_val = val
                    best_comb = new_comb
        
        return best_comb
    
    def _init_solve(self):

        store_ids = list(self.price_matrix.columns)
        product_ids = [self.product_encoder.encode(i) for i in self.product_query]
        self.price_matrix = self.price_matrix.loc[product_ids, :]

        #print("Building catalog combinations...")
        #catalog_combinations = list(list(zip(element, product_ids))
            #for element in itertools.product(store_ids, repeat = len(product_ids)))
        catalog_combinations = self._backtrack([], 0)
        
        #print("Building store combinations...")
        store_combinations = list(itertools.permutations(store_ids))

        return catalog_combinations, store_combinations

    def solve(self):

        store_combinations, catalog_combinations = self._init_solve()

        #print("Getting best value...")
        best_comb = self._evaluate_bf_solutions(store_combinations, catalog_combinations)

        return best_comb
    
class QuasiBruteForceSolver(BruteForceSolver):

    """
    This solver searches a sample of all possible solutions
    """

    def solve(self, max_catalog_comb_sample=10000, max_store_comb_sample=1000):

        catalog_combinations, store_combinations = self._init_solve()
        catalog_combinations = random.sample(catalog_combinations,
                            min(len(catalog_combinations), max_catalog_comb_sample))
        store_combinations = random.sample(store_combinations,
                            min(len(store_combinations), max_store_comb_sample))

        best_comb = self._evaluate_bf_solutions(store_combinations, catalog_combinations)

        return best_comb
    