#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np
import tsplib95

from tabu_search import tabu_search

TSP_FILE = "./hk48.tsp"
RUNS = 30


def init_graph(fun_name, title):
    plt.xlabel("iterations")
    plt.ylabel("cost function")
    plt.grid(True)
    plt.title(title)


if __name__ == "__main__":
    problem = tsplib95.load(TSP_FILE)
    print(f"TSP: {problem.name}")
    print(f"Nodes: {len(list(problem.get_nodes()))}")
    init_graph(problem.name, f"tabu_search, {problem.name}, {RUNS} runs")
    cost_best_results = []
    cost_history_results = []
    distance_cache = {}
    total_distance_cache = {}
    for i in range(RUNS):
        cost_best, path_best, costs_history = tabu_search(problem, distance_cache, total_distance_cache)
        cost_best_results.append(cost_best)
        cost_history_results.append(costs_history)
        plt.plot(range(1, len(costs_history)+1), costs_history, linewidth=0.5)
        sys.stdout.write(f"\rRun {i+1}/{RUNS} completed.")
        sys.stdout.flush()
    print("")
    
    plt.savefig(f"tabu_search_{problem.name}_all.png")
    plt.clf()
    plt.cla()
    plt.close()

    average_run_history = np.mean(cost_history_results, axis=0)
    init_graph(problem.name, f"tabu_search, {problem.name}, average best run")
    plt.plot(range(1, len(average_run_history)+1), average_run_history, linewidth=1)
    plt.savefig(f"tabu_search_{problem.name}_average.png")
    plt.clf()
    plt.cla()
    plt.close()

    minimum = np.min(cost_best_results)
    maximum = np.max(cost_best_results)
    mean = np.mean(cost_best_results)
    median = np.median(cost_best_results)
    stddev = np.std(cost_best_results)

    print(f"tabu_search, {problem.name}: min={minimum}, max={maximum}, mean={mean}, median={median}, stddev={stddev}")
