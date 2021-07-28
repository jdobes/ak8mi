import numpy as np


def random_search(bounds, dimension, cost_function, max_fes):
    cost_best = np.Inf
    position_best = np.random.uniform(bounds[0], bounds[1], dimension)
    costs_history = []

    for _ in range(max_fes):
        position = np.random.uniform(bounds[0], bounds[1], dimension)
        cost = cost_function(position)
        if cost < cost_best:
            cost_best = cost
            position_best = position
        costs_history.append(cost_best)

    return cost_best, position_best, costs_history
