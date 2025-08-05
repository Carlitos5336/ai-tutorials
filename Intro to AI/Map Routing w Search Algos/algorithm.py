import heapq
from front.Folium import draw_location, empty_map

def uniform_cost_search(start_node, goal_node, adjacency_list):
    """
    Executes the Uniform Cost Search (UCS) algorithm to find the lowest-cost path
    from start_node to goal_node using the provided adjacency_list.

    Parameters:
    - start_node: tuple (lat, lon)
    - goal_node: tuple (lat, lon)
    - adjacency_list: dict with neighbors and edge costs

    Returns:
    - None. Draws the solution route using Folium.
    """

    frontier = []  # Priority queue of (node, cost)
    frontier_states = set()  # Set of nodes in frontier
    explored = set()  # Set of visited nodes
    parents = {start_node: start_node}
    costs = {start_node: 0}

    heapq.heappush(frontier, (0, start_node))  # (cost, node)
    frontier_states.add(start_node)

    while frontier:
        current_cost, current_node = heapq.heappop(frontier)
        frontier_states.discard(current_node)
        explored.add(current_node)

        if current_node == goal_node or current_node not in adjacency_list:
            # Reconstruct solution path
            solution_path = []
            n = current_node
            while parents[n] != n:
                solution_path.append(n)
                n = parents[n]
            solution_path.append(start_node)
            solution_path.reverse()

            print("Reached destination. Total cost:", costs[current_node])
            print("Solution path:", solution_path)

            draw_location(solution_path)
            break

        for neighbor, edge_cost in adjacency_list[current_node]:
            new_cost = costs[current_node] + edge_cost

            if neighbor not in frontier_states and neighbor not in explored:
                costs[neighbor] = new_cost
                heapq.heappush(frontier, (new_cost, neighbor))
                parents[neighbor] = current_node
                frontier_states.add(neighbor)

            elif new_cost < costs.get(neighbor, float('inf')):
                costs[neighbor] = new_cost
                parents[neighbor] = current_node
                # Update priority queue (inefficiently, unless using better heap)
                heapq.heappush(frontier, (new_cost, neighbor))
