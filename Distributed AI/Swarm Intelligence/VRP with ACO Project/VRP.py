import itertools, sys, time
import numpy as np
import pandas as pd
from collections import Counter

class ClarkWrightSolver():

  def __init__(self):
    self.savings = None
    self.nodes = None
    self.distance_matrix = None

  def calculate_savings(self):
    # calculate savings for each link
    savings = dict()
    for r in self.distance_matrix.index:
        for c in self.distance_matrix.columns:
            if int(c) != int(r):
                a = max(int(r), int(c))
                b = min(int(r), int(c))
                key = '(' + str(a) + ',' + str(b) + ')'
                savings[key] = self.nodes['d0'][int(r)] + self.nodes['d0'][int(c)] - self.distance_matrix[c][r]

    # put savings in a pandas dataframe, and sort by descending
    self.savings = pd.DataFrame.from_dict(savings, orient = 'index')
    self.savings.rename(columns = {0:'saving'}, inplace = True)
    self.savings.sort_values(by = ['saving'], ascending = False, inplace = True)

  # convert link string to link list to handle saving's key, i.e. str(10, 6) to (10, 6)
  def get_node(self, link):
      link = link[1:]
      link = link[:-1]
      nodes = link.split(',')
      return [int(nodes[0]), int(nodes[1])]

  # determine if a node is interior to a route
  @staticmethod
  def interior(node, route):
      try:
          i = route.index(node)
          # adjacent to depot, not interior
          if i == 0 or i == (len(route) - 1):
              label = False
          else:
              label = True
      except:
          label = False

      return label

  # merge two routes with a connection link
  @staticmethod
  def merge(route0, route1, link):
      if route0.index(link[0]) != (len(route0) - 1):
          route0.reverse()

      if route1.index(link[1]) != 0:
          route1.reverse()

      return route0 + route1

  # sum up to obtain the total passengers belonging to a route
  def sum_cap(self, route):
      sum_cap = 0
      for node in route:
          sum_cap += self.nodes.demand[node]
      return sum_cap

  # determine 4 things:
  # 1. if the link in any route in routes -> determined by if count_in > 0
  # 2. if yes, which node is in the route -> returned to node_sel
  # 3. if yes, which route is the node belongs to -> returned to route id: i_route
  # 4. are both of the nodes in the same route? -> overlap = 1, yes; otherwise, no
  @staticmethod
  def which_route(link, routes):
      # assume nodes are not in any route
      node_sel = list()
      i_route = [-1, -1]
      count_in = 0

      for route in routes:
          for node in link:
              try:
                  route.index(node)
                  i_route[count_in] = routes.index(route)
                  node_sel.append(node)
                  count_in += 1
              except:
                  pass

      if i_route[0] == i_route[1]:
          overlap = 1
      else:
          overlap = 0

      return node_sel, count_in, i_route, overlap

  def solve(self, nodes, distance_matrix, cap):
    
    nodes["d0"] = distance_matrix.iloc[0]
    self.nodes = nodes[["d0", "demand"]]
    self.distance_matrix = distance_matrix.iloc[1:, 1:]
    self.calculate_savings()

    # create empty routes
    routes = list()

    # if there is any remaining customer to be served
    remaining = True

    # record steps
    step = 0

    # get a list of nodes, excluding the depot
    node_list = list(self.nodes.index)
    node_list.remove(0)

    # run through each link in the saving list
    for link in self.savings.index:
        step += 1
        if remaining:

            #print('step ', step, ':')

            link = self.get_node(link)
            node_sel, num_in, i_route, overlap = self.which_route(link, routes)

            # condition a. Either, neither i nor j have already been assigned to a route,
            # ...in which case a new route is initiated including both i and j.
            if num_in == 0:
                if self.sum_cap(link) <= cap:
                    routes.append(link)
                    node_list.remove(link[0])
                    node_list.remove(link[1])
                    #print('\t','Link ', link, ' fulfills criteria a), so it is created as a new route')
                else:
                    pass
                    #print('\t','Though Link ', link, ' fulfills criteria a), it exceeds maximum load, so skip this link.')

            # condition b. Or, exactly one of the two nodes (i or j) has already been included
            # ...in an existing route and that point is not interior to that route
            # ...(a point is interior to a route if it is not adjacent to the depot D in the order of traversal of nodes),
            # ...in which case the link (i, j) is added to that same route.
            elif num_in == 1:
                n_sel = node_sel[0]
                i_rt = i_route[0]
                position = routes[i_rt].index(n_sel)
                link_temp = link.copy()
                link_temp.remove(n_sel)
                node = link_temp[0]

                cond1 = (not self.interior(n_sel, routes[i_rt]))
                cond2 = (self.sum_cap(routes[i_rt] + [node]) <= cap)

                if cond1:
                    if cond2:
                        #print('\t','Link ', link, ' fulfills criteria b), so a new node is added to route ', routes[i_rt], '.')
                        if position == 0:
                            routes[i_rt].insert(0, node)
                        else:
                            routes[i_rt].append(node)
                        node_list.remove(node)
                    else:
                        #print('\t','Though Link ', link, ' fulfills criteria b), it exceeds maximum load, so skip this link.')
                        continue
                else:
                    #print('\t','For Link ', link, ', node ', n_sel, ' is interior to route ', routes[i_rt], ', so skip this link')
                    continue

            # condition c. Or, both i and j have already been included in two different existing routes
            # ...and neither point is interior to its route, in which case the two routes are merged.
            else:
                if overlap == 0:
                    cond1 = (not self.interior(node_sel[0], routes[i_route[0]]))
                    cond2 = (not self.interior(node_sel[1], routes[i_route[1]]))
                    cond3 = (self.sum_cap(routes[i_route[0]] + routes[i_route[1]]) <= cap)

                    if cond1 and cond2:
                        if cond3:
                            route_temp = self.merge(routes[i_route[0]], routes[i_route[1]], node_sel)
                            temp1 = routes[i_route[0]]
                            temp2 = routes[i_route[1]]
                            routes.remove(temp1)
                            routes.remove(temp2)
                            routes.append(route_temp)
                            try:
                                node_list.remove(link[0])
                                node_list.remove(link[1])
                            except:
                                #print('\t', f"Node {link[0]} or {link[1]} has been removed in a previous step.")
                                pass
                            #print('\t','Link ', link, ' fulfills criteria c), so route ', temp1, ' and route ', temp2, ' are merged')
                        else:
                            #print('\t','Though Link ', link, ' fulfills criteria c), it exceeds maximum load, so skip this link.')
                            continue
                    else:
                        #print('\t','For link ', link, ', Two nodes are found in two different routes, but not all the nodes fulfill interior requirement, so skip this link')
                        continue
                else:
                    #print('\t','Link ', link, ' is already included in the routes')
                    continue

            for route in routes:
                pass
                #print('\t','route: ', route, ' with load ', self.sum_cap(route))
        else:
            #print('-------')
            #print('All nodes are included in the routes, algorithm closed')
            break

        remaining = bool(len(node_list) > 0)

    # check if any node is left, assign to a unique route
    for node_o in node_list:
        routes.append([node_o])

    # add depot to the routes
    #for route in routes:
        #route.insert(0,0)
        #route.append(0)

    #print('------')
    #print('Routes found are: ')

    return routes, self.solution_length(routes), len(routes)
  
  def sum_length(self, route):
      sum_cap = 0
      for i in range(len(route) - 1):
          sum_cap += self.distance_matrix[route[i]][route[i+1]]
      sum_cap += self.nodes["d0"][route[0]] + self.nodes["d0"][route[-1]]
      return sum_cap

  def solution_length(self, routes):
    return sum([self.sum_length(route) for route in routes])

class BruteForceSolver():

  def __init__(self):
    self.nodes = None
    self.distance_matrix = None

  def all_partitions(self, collection):
    if len(collection) == 1:
      yield [collection]
      return
    first = collection[0]
    for smaller in self.all_partitions(collection[1:]):
      for n, subset in enumerate(smaller):
        yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
      yield [[first]] + smaller

  @staticmethod
  def get_permutations(data):
    return list(itertools.permutations(data))

  def sum_cap(self, route):
      sum_cap = 0
      for node in route:
          sum_cap += self.nodes.demand[node]
      return sum_cap

  def sum_length(self, route):
      sum_cap = 0
      for i in range(len(route) - 1):
          sum_cap += self.distance_matrix[route[i]][route[i+1]]
      sum_cap += self.nodes["d0"][route[0]] + self.nodes["d0"][route[-1]]
      return sum_cap

  def solution_length(self, routes):
    return sum([self.sum_length(route) for route in routes])

  def solve(self, nodes, distance_matrix, cap):
    nodes["d0"] = distance_matrix.iloc[0]
    self.nodes = nodes[["d0", "demand"]]
    self.distance_matrix = distance_matrix.iloc[1:, 1:]
    best_part = None
    best_val = np.inf
    for partition in list(self.all_partitions([i for i in self.distance_matrix.index])):
      if all([self.sum_cap(route) <= cap for route in partition]):
        new_val = 0
        new_partition = []
        for route in partition:
          perms = self.get_permutations(route)
          best_min = np.inf
          best_perm = None
          for perm in perms:
            new_min = self.sum_length(perm)
            if new_min < best_min:
              best_min = new_min
              best_perm = perm
          new_val += best_min
          new_partition.append(list(best_perm))
        if new_val < best_val:
          best_part = new_partition
          best_val = new_val
    return best_part, self.solution_length(best_part), len(best_part)

from collections import Counter

class ACOSolver():

  def __init__(self, alpha=1, beta=1, rho=0.5, max_iters=100, ants=20):
    self.nodes = None
    self.distance_matrix = None
    self.alpha = alpha
    self.beta = beta
    self.rho = rho
    self.max_iters = max_iters
    self.ants = ants

  def sum_cap(self, route):
      sum_cap = 0
      for node in route:
          sum_cap += self.nodes.demand[node]
      return sum_cap

  def sum_length(self, route):
      sum_cap = 0
      for i in range(len(route) - 1):
          sum_cap += self.distance_matrix[route[i]][route[i+1]]
      sum_cap += self.nodes["d0"][route[0]] + self.nodes["d0"][route[-1]]
      return sum_cap

  def solution_length(self, routes):
    return sum([self.sum_length(route) for route in routes])

  def solve(self, nodes, distance_matrix, cap):
    nodes["d0"] = distance_matrix.iloc[0]
    self.nodes = nodes[["d0", "demand"]]
    self.distance_matrix = distance_matrix
    raw_path = self.__capacitated_aco(cap)
    best_path = []
    for path in raw_path:
       best_path.append(self.__local_aco(path))
    return best_path, self.solution_length(best_path), len(best_path)

  def __local_aco(self, tpath):

    d = np.array(self.distance_matrix)
    d[np.identity(d.shape[0], dtype="bool")] = np.inf
    tau = np.ones(d.shape)
    tau_d = np.zeros(d.shape)
    neta = 1/d
    neta[neta == np.inf] = 1e6
    initial_state = 0

    for iter in range(self.max_iters):
        paths = []
        for ant in range(self.ants):
            path = []
            unvisited = [0] + tpath.copy()
            idx = initial_state
            unvisited.remove(idx)
            path.append(idx)
            while len(unvisited) != 0:
                w = tau**self.alpha * neta**self.beta
                w = w[unvisited]
                p = w/sum(w)
                idx = np.random.choice(unvisited, p=p[:,idx])
                unvisited.remove(idx)
                path.append(idx)
            path.append(initial_state)
            dt = 0
            for i in range(len(path)-1):
                dt += d[path[i], path[i+1]]
            dp = 1/dt
            for i in range(len(path)-1):
                tau_d[path[i], path[i+1]] += dp
                tau_d[path[i+1], path[i]] += dp
            paths.append(path)
            #print(f"Ant {ant} path: {path}. Total distance: {dt}. Total pheromones: {dp}")
        tau = (1 - self.rho) * tau + tau_d
    return [int(j) for j in Counter(["|".join([str(i) for i in path]) for path in paths]).most_common(1)[0][0].split("|")][1:-1]

  def __capacitated_aco(self, cap):

    d = np.array(self.distance_matrix)
    d[np.identity(d.shape[0], dtype="bool")] = np.inf
    tau = np.ones(d.shape)
    tau_d = np.zeros(d.shape)
    neta = 1/d
    neta[neta == np.inf] = 1e6
    initial_state = 0
    n_ants = max(self.ants, len(self.nodes))

    for iter in range(self.max_iters):
        paths = []
        global_unvisited = [i for i in range(1, d.shape[0])]
        for ant in range(n_ants):
            capacity = cap
            path = []
            unvisited = [0] + [i for i in global_unvisited]
            idx = initial_state
            unvisited.remove(idx)
            path.append(idx)
            while len(unvisited) != 0:
                w = tau**self.alpha * neta**self.beta
                unv_filter = [i for i in unvisited if capacity - self.nodes.loc[i, "demand"] >= 0]
                if len(unv_filter) == 0: break
                w = w[unv_filter]
                p = w/sum(w)
                idx = np.random.choice(unv_filter, p=p[:,idx])
                unvisited.remove(idx)
                if idx != 0: global_unvisited.remove(idx)
                path.append(idx)
                capacity -= self.nodes.loc[idx, "demand"]
                if len(global_unvisited) == 0:
                    break
                    #global_unvisited = [i for i in range(1, d.shape[0])]
            path.append(initial_state)
            dt = 0
            ct = 0
            for i in range(len(path)-1):
                dt += d[path[i], path[i+1]]
                ct += self.nodes.loc[i, "demand"]
            dp = 1/dt
            if dt == 0: continue
            for i in range(len(path)-1):
                tau_d[path[i], path[i+1]] += dp
                tau_d[path[i+1], path[i]] += dp
            paths.append(path)
        tau = (1 - self.rho) * tau + tau_d
    return [i[1:-1] for i in paths if len(i) != 2]