import random

ITERATIONS = 250
TABU_LIST_SIZE = 20


def get_distance(problem, distance_cache, a, b):
    if (a, b) in distance_cache:
        return distance_cache[(a, b)]
    
    if (b, a) in distance_cache:
        return distance_cache[(b, a)]
    
    distance = problem.get_weight(a, b)
    distance_cache[(a, b)] = distance
    return distance


def calc_cost(problem, path, distance_cache, total_distance_cache):
    cached_cost = total_distance_cache.get(tuple(path))
    if cached_cost is not None:
        return cached_cost

    cost = 0
    nodes_count = len(path)
    for i in range(nodes_count - 1):
        cost += get_distance(problem, distance_cache, path[i], path[i+1])
    cost += get_distance(problem, distance_cache, path[nodes_count - 1], path[0])
    total_distance_cache[tuple(path)] = cost
    return cost


def tabu_search(problem, distance_cache, total_distance_cache):
    path_best = list(problem.get_nodes())
    random.shuffle(path_best)
    nodes_count = len(path_best)
    cost_best = calc_cost(problem, path_best, distance_cache, total_distance_cache)
    costs_history = []
    tabu_list = []

    path = path_best[:]
    for _ in range(ITERATIONS):
        path_local = path[:]
        cost_local = calc_cost(problem, path_local, distance_cache, total_distance_cache)
        transformation_local = None
        # evaluate for all possible neighbor swaps
        path_candidate = path_local[:]
        for i in range(nodes_count):
            for j in range(i + 1, nodes_count):
                path_candidate[i], path_candidate[j] = path_candidate[j], path_candidate[i]
                cost_candidate = calc_cost(problem, path_candidate, distance_cache, total_distance_cache)
                if cost_candidate < cost_local and (((i, j) not in tabu_list and (j, i) not in tabu_list) or (cost_candidate < cost_best)):
                    path_local = path_candidate[:]
                    cost_local = cost_candidate
                    transformation_local = (i, j)
                path_candidate[j], path_candidate[i] = path_candidate[i], path_candidate[j]

        if cost_local < cost_best:
            cost_best = cost_local
            path_best = path_local[:]

        if transformation_local:  # add to tabu list
            i, j = transformation_local
            tabu_list.append((j, i))
            if len(tabu_list) > TABU_LIST_SIZE:
                tabu_list = tabu_list[1:]

        path = path_local[:]
        costs_history.append(cost_best)

    return cost_best, path_best, costs_history
