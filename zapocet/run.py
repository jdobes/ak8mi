#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

from random_search import random_search
from local_search import local_search
from simulated_annealing import simulated_annealing


def first_dejong(x):
    return np.sum(x**2)


def second_dejong(x):
    return np.sum(100 * (x[:-1] ** 2 - x[1:]) ** 2 + (1 - x[:-1]) ** 2)


def schweffel(x):
    return np.sum((-1) * x * np.sin(np.sqrt(np.abs(x))))


RUNS = 30
MAX_FES = 10000
DIMENSION = 10

COST_FUNCTIONS = {
    "first_dejong": first_dejong,
    "second_dejong": second_dejong,
    "schweffel": schweffel
}

BOUNDS = {
    "first_dejong": [-5, 5],
    "second_dejong": [-5, 5],
    "schweffel": [-500, 500]
}

GRAPH_BOUNDS = {
    "first_dejong": (0, MAX_FES, 0, 100),
    "second_dejong": (0, MAX_FES, 0, 10000),
    "schweffel": (0, MAX_FES, -419*DIMENSION, 0)
}

SEARCH_ALGORITHMS = {
    "random_search": random_search,
    "local_search": local_search,
    "simulated_annealing": simulated_annealing
}

def init_graph(fun_name, title):
    plt.xlabel("FES")
    plt.ylabel("cost function")
    plt.grid(True)
    plt.axis(GRAPH_BOUNDS[f"{fun_name}"])
    plt.title(title)

if __name__ == "__main__":
    for cost_function_name, cost_function in COST_FUNCTIONS.items():
        comparison = {}
        for search_algorithm_name, search_algorithm in SEARCH_ALGORITHMS.items():
            init_graph(cost_function_name, f"{search_algorithm_name}, {cost_function_name}, {DIMENSION} dimensions, {RUNS} runs")
            cost_best_results = []
            cost_history_results = []
            for _ in range(RUNS):
                cost_best, position_best, costs_history = search_algorithm(BOUNDS[cost_function_name], DIMENSION, cost_function, MAX_FES)
                cost_best_results.append(cost_best)
                cost_history_results.append(costs_history)
                #print(f"{search_algorithm_name}, {cost_function_name}, {DIMENSION} dimensions: cost_best={cost_best}, position_best={position_best}")
                plt.plot(range(1, len(costs_history)+1), costs_history, linewidth=0.5)

            plt.savefig(f"{search_algorithm_name}_{cost_function_name}_all.png")
            plt.clf()
            plt.cla()
            plt.close()

            minimum = np.min(cost_best_results)
            maximum = np.max(cost_best_results)
            mean = np.mean(cost_best_results)
            median = np.median(cost_best_results)
            stddev = np.std(cost_best_results)

            average_run_history = np.mean(cost_history_results, axis=0)
            comparison[search_algorithm_name] = average_run_history
            init_graph(cost_function_name, f"{search_algorithm_name}, {cost_function_name}, {DIMENSION} dimensions, average best run")
            plt.plot(range(1, len(average_run_history)+1), average_run_history, linewidth=1)
            plt.savefig(f"{search_algorithm_name}_{cost_function_name}_average.png")
            plt.clf()
            plt.cla()
            plt.close()

            print(f"{search_algorithm_name}, {cost_function_name}, {DIMENSION} dimensions: min={minimum}, max={maximum}, mean={mean}, median={median}, stddev={stddev}")
        
        init_graph(cost_function_name, f"{cost_function_name}, {DIMENSION} dimensions, average best run comparison")
        for search_algorithm_name, average_run_history in comparison.items():
            plt.plot(range(1, len(average_run_history)+1), average_run_history, linewidth=1, label=search_algorithm_name)
        plt.legend()
        plt.savefig(f"{cost_function_name}_comparison.png")
        plt.clf()
        plt.cla()
        plt.close()

