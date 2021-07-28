import numpy as np

NEIGHBORHOOD_SIZE = 10
LOCAL_SIZE_PERCENTAGE = 0.1


def random_neighbor(dimension, bounds, local_area_length, position_original):
    position = []
    for i in range(dimension):
        coordinate = np.random.normal(position_original[i], local_area_length / 2)
        # ensure boundaries
        while not coordinate >= bounds[0] or not coordinate <= bounds[1]:
            coordinate = np.random.normal(position_original[i], local_area_length / 2)
        position.append(coordinate)
    position = np.asarray(position)
    return position


def local_search(bounds, dimension, cost_function, max_fes):
    bounds_length = bounds[1] - bounds[0]
    local_area_length = bounds_length * LOCAL_SIZE_PERCENTAGE

    cost_best = np.Inf
    position_best = np.random.uniform(bounds[0], bounds[1], dimension)
    costs_history = []

    for _ in range(int(max_fes / NEIGHBORHOOD_SIZE)):
        found_better = False
        # generate points in local area
        position_original = position_best
        for _ in range(NEIGHBORHOOD_SIZE):
            position = random_neighbor(dimension, bounds, local_area_length, position_original)
            cost = cost_function(position)
            if cost < cost_best:
                cost_best = cost
                position_best = position
                found_better = True
            costs_history.append(cost_best)
        if not found_better:
            break
    
    # fill missing values to target FES
    last_val = costs_history[-1]
    for _ in range(max_fes - len(costs_history)):
        costs_history.append(last_val)

    return cost_best, position_best, costs_history
